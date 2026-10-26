"""Prepare event-local SS-reference databases; no cross-event stacking."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.signal import correlate, find_peaks
from .batch import read_plan, event_cache, task_destination, run_tasks
from .pipeline import prepare, load_preprocessed, grid, dump_json
from .stacking import sample_trace


ALIGNMENT_DEFAULTS = dict(template_window_s=[-40,40], max_lag_s=30,
    minimum_correlation=.7, minimum_peak_margin=.05, peak_separation_s=10,
    reference_candidate_count=20, preferred_reference_distance_degree=[100,140])


def alignment_settings(configuration):
    settings=dict(ALIGNMENT_DEFAULTS,**configuration.get('event_reference_alignment',{}))
    if set(settings)-set(ALIGNMENT_DEFAULTS):
        raise ValueError('Unknown event-reference alignment setting')
    lo,hi=settings['template_window_s']; dlo,dhi=settings['preferred_reference_distance_degree']
    if (not np.isfinite([lo,hi,dlo,dhi,settings['max_lag_s'],settings['minimum_correlation'],
                         settings['minimum_peak_margin'],settings['peak_separation_s']]).all()
            or not lo<0<hi or not 0<=dlo<dhi<=180 or settings['max_lag_s']<=0
            or not 0<settings['minimum_correlation']<=1 or not 0<=settings['minimum_peak_margin']<=1
            or settings['peak_separation_s']<=0 or settings['reference_candidate_count']<1):
        raise ValueError('Invalid event-reference alignment settings')
    return settings


def template_lag(tau, trace, template_time, template, max_lag_s, separation_s):
    """Positive lag queries target(t+lag); positive shifts move arrivals earlier."""
    dt=float(tau[1]-tau[0]); half=int(np.floor(max_lag_s/dt))
    if half<1: raise ValueError('Lag range needs at least one sample on each side')
    time=template_time[0]+np.arange(len(template)+2*half)*dt-half*dt
    search=sample_trace(tau,trace,time)
    ref=np.asarray(template,dtype=float); ref=ref-ref.mean()
    if not np.isfinite(search).all() or not np.isfinite(ref).all() or np.dot(ref,ref)<=0:
        raise ValueError('Insufficient finite template/search energy or support')
    n=len(ref); prefix=np.r_[0,np.cumsum(search)]; square=np.r_[0,np.cumsum(search**2)]
    energy=square[n:]-square[:-n]-(prefix[n:]-prefix[:-n])**2/n
    den=np.sqrt(np.maximum(energy,0)*np.dot(ref,ref))
    coefficients=np.divide(correlate(search,ref,mode='valid',method='direct'),den,
                           out=np.full(2*half+1,np.nan),where=den>0)
    if not np.isfinite(coefficients).any(): raise ValueError('No valid correlation samples')
    best=int(np.nanargmax(coefficients)); boundary=best in (0,len(coefficients)-1)
    delta=0.
    if not boundary:
        a,b,c=coefficients[best-1:best+2]; curvature=a-2*b+c
        if np.isfinite(curvature) and curvature<0: delta=float(np.clip(.5*(a-c)/curvature,-.5,.5))
    lag=(best-half+delta)*dt
    alternate=find_peaks(np.where(np.isfinite(coefficients),coefficients,-np.inf))[0]
    alternate=alternate[abs(alternate-best)*dt>=separation_s]
    alternate=alternate[coefficients[alternate]>0]
    secondary=float(np.max(coefficients[alternate])) if len(alternate) else None
    peak=float(np.clip(coefficients[best],-1,1))
    return dict(lag_s=float(lag),correlation=peak,lag_at_boundary=boundary,
                secondary_correlation=secondary,peak_margin=peak-secondary if secondary is not None else None)


def choose_reference(tau,data,rows,settings):
    dt=float(tau[1]-tau[0]); lo,hi=settings['template_window_s']
    template_time=np.arange(lo,hi+dt/2,dt)
    pulse=np.array([sample_trace(tau,u,template_time) for u in data])
    noise_time=np.r_[np.arange(-180,-80,dt),np.arange(80,180,dt)]
    noise=np.array([sample_trace(tau,u,noise_time) for u in data])
    signal_power=np.mean(pulse**2,axis=1); noise_power=np.mean(noise**2,axis=1)
    quality=np.divide(signal_power,noise_power,out=np.full(len(data),np.inf),where=noise_power>0)
    shape=pulse-pulse.mean(axis=1,keepdims=True); norms=np.linalg.norm(shape,axis=1)
    if not np.all(norms>0) or not np.isfinite(shape).all():
        raise ValueError('Nonfinite or constant candidate SS window')
    shape=shape/norms[:,None]
    low,high=settings['preferred_reference_distance_degree']
    pool=np.array([i for i,r in enumerate(rows) if low<=r['distance_degree']<=high],dtype=int)
    if not len(pool): pool=np.arange(len(rows))
    candidates=pool[np.argsort(-quality[pool],kind='stable')[:settings['reference_candidate_count']]]
    medians=np.median(shape[candidates]@shape.T,axis=1)
    best=int(candidates[int(np.argmax(medians))])
    return best,template_time,pulse[best],dict(candidate_indices=candidates.tolist(),
        candidate_median_correlations=medians.tolist(),reference_ss_to_coda_power_ratio=
        float(quality[best]) if np.isfinite(quality[best]) else None,
        definition='Among high SS-to-coda-power candidates, choose highest median zero-lag waveform correlation to event stations.')


def align_event(tau,data,rows,configuration):
    if not len(rows) or len(data)!=len(rows): raise ValueError('Event has no usable station waveforms')
    if len({r['event_id'] for r in rows})!=1: raise ValueError('Event reference cannot span multiple events')
    settings=alignment_settings(configuration)
    ref,template_time,template,selection=choose_reference(tau,data,rows,settings)
    final_tau=grid(configuration['relative_time_s']); accepted=[]; rejected=[]; aligned=[]; qc=[]
    for i,(u,original) in enumerate(zip(data,rows)):
        row=dict(original)
        fit=template_lag(tau,u,template_time,template,settings['max_lag_s'],settings['peak_separation_s'])
        if i==ref: fit.update(lag_s=0.,correlation=1.,lag_at_boundary=False)
        reason=None
        if fit['lag_at_boundary']: reason='Event-reference SS lag is at search boundary'
        elif fit['correlation']<settings['minimum_correlation']: reason='Event-reference SS correlation below threshold'
        elif i!=ref and fit['peak_margin'] is not None and fit['peak_margin']<settings['minimum_peak_margin']:
            reason='Event-reference SS correlation peak is ambiguous'
        values=sample_trace(tau,u,final_tau+fit['lag_s'])
        if not np.isfinite(values).all(): reason='Aligned output window exceeds reliable waveform support'
        row.update(coarse_measured_ss_s=row['measured_ss_s'],coarse_alignment_residual_s=row['alignment_residual_s'],
                   reference_trace_id=rows[ref]['trace_id'],reference_station=rows[ref]['station'],
                   event_reference_alignment=fit,alignment_method='event_local_SS_reference')
        row['measured_ss_s']=row['coarse_measured_ss_s']+fit['lag_s']
        row['alignment_residual_s']=row['measured_ss_s']-row['predicted_ss_s']
        qc.append(dict(trace_id=row['trace_id'],station=row['station'],accepted=reason is None,reason=reason,**fit))
        if reason:
            row.update(status='rejected',reason=reason,alignment_applied=False,
                       measured_ss_s=row['coarse_measured_ss_s'],alignment_residual_s=row['coarse_alignment_residual_s'])
            rejected.append(row)
        else:
            row.update(status='accepted',alignment_applied=True)
            accepted.append(row); aligned.append(values)
    matrix=np.asarray(aligned).reshape(len(aligned),len(final_tau))
    ref_row=next(i for i,r in enumerate(accepted) if r['trace_id']==rows[ref]['trace_id'])
    reference=dict(event_id=rows[ref]['event_id'],trace_id=rows[ref]['trace_id'],station=rows[ref]['station'],
                   accepted_row_index=ref_row,template_time_s=template_time.tolist(),template_samples=template.tolist(),
                   selected_coarse_row_index=ref,settings=settings,selection=selection,
                   time_origin_note='Zero is the chosen event SS reference peak; it is not an absolute theoretical arrival.')
    return final_tau,matrix,accepted,rejected,reference,qc


def prepare_event_database(plan_path,root,index):
    plan=read_plan(plan_path,'events-v1')
    if not 0<=index<len(plan['tasks']): raise ValueError('Event index out of range')
    task=plan['tasks'][index]; cfg=dict(plan['configuration'],event_id=task['event_id'])
    settings=alignment_settings(cfg); final=grid(cfg['relative_time_s'])
    source=Path(task['source']); stat=source.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(task['source_size_bytes'],task['source_mtime_ns']):
        raise ValueError('ASDF input changed since event planning')
    destination=event_cache(root,index)
    with task_destination(destination.parent,destination.name,{'plan':plan['fingerprint'],'task_index':index}) as work:
        if work is None: return dict(task_index=index,status='cached',output=str(destination))
        # Save extra coarse waveform support for fractional shifts, then crop once.
        margin=settings['max_lag_s']+2*float(final[1]-final[0])
        coarse_cfg=dict(cfg,relative_time_s=[float(final[0]-margin),float(final[-1]+margin),float(final[1]-final[0])])
        with tempfile.TemporaryDirectory(prefix='coarse-',dir=str(work)) as temporary:
            coarse=Path(temporary)
            prepare(source,coarse_cfg,coarse,network=plan['network'],cache_format='npy')
            tau,data=load_preprocessed(coarse)
            metadata=json.loads((coarse/'traces.json').read_text())
            prov=json.loads((coarse/'provenance.json').read_text())
            final_tau,matrix,accepted,rejected,reference,qc=align_event(tau,data,metadata['accepted'],cfg)
            unresolved_ids={r['trace_id'] for r in rejected}
            unresolved=[i for i,r in enumerate(metadata['accepted']) if r['trace_id'] in unresolved_ids]
            np.save(work/'unaligned_tau.npy',tau,allow_pickle=False)
            np.save(work/'unaligned_waveforms.npy',np.asarray(data)[unresolved],allow_pickle=False)
            dump_json(work/'unaligned_traces.json',rejected)
            del data
        np.save(work/'tau.npy',final_tau,allow_pickle=False)
        np.save(work/'waveforms.npy',matrix,allow_pickle=False)
        np.save(work/'reference.npy',matrix[reference['accepted_row_index']],allow_pickle=False)
        dump_json(work/'reference.json',reference)
        dump_json(work/'alignment_qc.json',qc)
        all_rejected=metadata['rejected']+rejected
        dump_json(work/'traces.json',dict(accepted=accepted,rejected=all_rejected))
        prov.update(configuration=cfg,alignment_method='event_local_SS_reference',
                    configuration_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                    coarse_relative_time_s=coarse_cfg['relative_time_s'],reference_trace_id=reference['trace_id'],
                    alignment_database_version=1,
                    lag_convention='Output u_i(t)=coarse_u_i(t+lag); the same shift applies to the entire waveform.',
                    ss_amplitude_note='Individual coarse SS peak amplitude normalization and whole-trace polarity retained; no extra amplitude scaling after correlation.')
        dump_json(work/'provenance.json',prov)
        summary=dict(event_id=task['event_id'],accepted=len(accepted),rejected=len(all_rejected),
                     reference_station=reference['station'],reference_trace_id=reference['trace_id'],
                     correlation_rejections=dict(Counter(r['reason'] for r in rejected)),
                     preprocessing_rejections=dict(Counter(r['reason'] for r in metadata['rejected'])))
        dump_json(work/'summary.json',summary)
    return dict(task_index=index,status='complete',output=str(destination),**summary)


def index_database(plan_path,root):
    plan=read_plan(plan_path,'events-v1'); root=Path(root); entries=[]; pending=[]
    for task in plan['tasks']:
        cache=event_cache(root,task['index'])
        if not (cache/'complete.json').exists():
            pending.append(task['index']); continue
        if json.loads((cache/'complete.json').read_text())!={'plan':plan['fingerprint'],'task_index':task['index']}:
            raise ValueError('Database cache belongs to another plan')
        summary=json.loads((cache/'summary.json').read_text())
        entries.append(dict(task_index=task['index'],event_id=task['event_id'],cache=str(cache.resolve()),
            reference_trace_id=summary['reference_trace_id'],reference_station=summary['reference_station'],
            accepted=summary['accepted'],rejected=summary['rejected']))
    root.mkdir(parents=True,exist_ok=True)
    dump_json(root/'database_index.json',dict(event_plan=str(Path(plan_path).resolve()),plan_fingerprint=plan['fingerprint'],
        events=entries,pending_event_tasks=pending,alignment_method='one_SS_reference_per_event',
        time_window_s=plan['configuration']['relative_time_s'],cross_event_stacking=False))
    return dict(database_index=str(root/'database_index.json'),completed=len(entries),pending=len(pending))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task-index',type=int)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--index-only',action='store_true',help='Summarize existing completed events without processing')
    args=parser.parse_args(); plan=read_plan(args.plan,'events-v1')
    if 'event_reference_alignment' not in plan['configuration']:
        parser.error('Use an event-reference database plan/config, not an existing peak-aligned run')
    indices=range(len(plan['tasks'])) if args.task_index is None else [args.task_index]
    if not args.index_only:
        run_tasks(prepare_event_database,[(args.plan.resolve(),args.root.resolve(),i) for i in indices],args.workers)
    if args.task_index is None or args.index_only:
        print(json.dumps(index_database(args.plan,args.root)),flush=True)


if __name__=='__main__': main()
