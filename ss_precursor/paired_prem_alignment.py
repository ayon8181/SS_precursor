"""Trial GLAD alignment against paired, already aligned PREM SS waveforms."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import correlate, find_peaks
from .distance_section import load_global, match_global, distance_stack
from .pipeline import dump_json
from .stacking import sample_trace


def absolute_template_lag(tau, target, template_time, template, max_lag=30., separation=10.):
    """Maximize absolute correlation for timing; never multiply data by its sign."""
    dt=float(tau[1]-tau[0]); half=int(np.floor(max_lag/dt))
    if half<1: raise ValueError('Lag search needs at least one sample')
    query=template_time[0]+np.arange(len(template)+2*half)*dt-half*dt
    search=sample_trace(tau,target,query)
    ref=np.asarray(template,dtype=float); ref=ref-ref.mean(); n=len(ref)
    if not np.isfinite(search).all() or not np.isfinite(ref).all() or np.dot(ref,ref)<=0:
        raise ValueError('Nonfinite or constant SS template/search')
    prefix=np.r_[0.,np.cumsum(search)]; squared=np.r_[0.,np.cumsum(search**2)]
    energy=squared[n:]-squared[:-n]-(prefix[n:]-prefix[:-n])**2/n
    den=np.sqrt(np.maximum(energy,0)*np.dot(ref,ref))
    coefficients=np.divide(correlate(search,ref,mode='valid',method='direct'),den,
                           out=np.full(2*half+1,np.nan),where=den>0)
    if not np.isfinite(coefficients).any():raise ValueError('No finite correlation')
    magnitude=np.abs(coefficients); best=int(np.nanargmax(magnitude))
    boundary=best in (0,len(coefficients)-1); delta=0.
    if not boundary:
        a,b,c=magnitude[best-1:best+2]; curvature=a-2*b+c
        if np.isfinite(curvature) and curvature<0:delta=float(np.clip(.5*(a-c)/curvature,-.5,.5))
    lag=(best-half+delta)*dt
    alternatives=find_peaks(np.where(np.isfinite(magnitude),magnitude,-np.inf))[0]
    alternatives=alternatives[np.abs(alternatives-best)*dt>=separation]
    secondary=float(np.max(magnitude[alternatives])) if len(alternatives) else None
    score=float(np.clip(magnitude[best],0,1))
    aligned=sample_trace(tau,target,template_time+lag)
    after=float(np.corrcoef(aligned,template)[0,1])
    return dict(lag_s=float(lag),absolute_correlation=score,
                signed_correlation_before=float(coefficients[half]),signed_correlation_after=after,
                signed_correlation_at_grid_peak=float(coefficients[best]),lag_at_boundary=boundary,
                secondary_absolute_correlation=secondary,
                peak_margin=score-secondary if secondary is not None else None,
                additional_polarity_multiplier=1.)


def save_section(directory, dataset, tau, data, rows, members, plan_path, plan, trial):
    directory.mkdir()
    edges=np.arange(100.,161.);distances=np.array([r['distance_degree'] for r in rows])
    stack,counts,support,columns=distance_stack(tau,data,distances,np.ones(len(rows)),edges)
    curves={}
    for phase in ['S^400S','S^670S']:
        v=np.full((3,60),np.nan)
        for col in range(60):
            values=[r['arrivals'][phase]['time_s']-r['predicted_ss_s'] for i,r in enumerate(rows)
                    if columns[i]==col and phase in r['arrivals']]
            if values:v[:,col]=np.percentile(values,[0,50,100])
        curves[phase]=v
    np.savez_compressed(directory/'distance_stack.npz',tau=tau,stack=stack,support=support,
        trace_counts=counts,distance_edges=edges,distance_centers=(edges[:-1]+edges[1:])/2,
        distance_window_bounds=np.column_stack([edges[:-1],edges[1:]]),
        **{p.replace('^','')+'_prediction_min_median_max_s':v for p,v in curves.items()})
    members = [dict(m, paired_prem_shift_applied_s=m['paired_prem_alignment']['lag_s']
                    if dataset == 'GLAD_after' else 0.) for m in members]
    dump_json(directory/'members.json',members)
    dump_json(directory/'provenance.json',dict(dataset=dataset,bin_plan=str(plan_path.resolve()),
        bin_plan_fingerprint=plan['fingerprint'],weighting='trace',overlapping_windows=False,
        global_stack=True,distance_range_degree=[100,160],period_band_s=plan['configuration']['period_band_s'],
        filter_order=plan['configuration']['filter_order'],trial=trial,
        polarity_note='PREM preprocessing multipliers retained; correlation sign never applied.',
        edge_note='No extrapolation: samples outside shifted cached support remain NaN and are excluded from means.',
        prediction_note='Original PREM theoretical precursor-minus-SS times; not waveform picks.'))
    return dict(stack=stack,counts=counts,support=support,curves=curves)


def plot_sections(output,tau,sections,number,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=np.r_[tau-.5,tau[-1]+.5];edges=np.arange(100.,161.);centers=(edges[:-1]+edges[1:])/2
    for zoom in [False,True]:
        fig,axes=plt.subplots(1,3,figsize=(17,8),sharey=True,constrained_layout=True)
        limit=.08 if zoom else .3
        for ax,(label,data) in zip(axes,sections):
            mesh=ax.pcolormesh(edges,times,data['stack'],cmap='RdBu',vmin=-limit,vmax=limit,rasterized=True)
            ax.set_facecolor('.85');ax.axhline(0,color='black',lw=.7)
            for phase,curve in data['curves'].items():
                ax.plot(centers,curve[1],'--',lw=.8,label=phase.replace('^','')+' prediction')
            ax.set(xlim=(100,160),ylim=(-300,-70) if zoom else (-500,300),
                   xlabel='Epicentral distance (degrees)',title=label)
            ax.legend(fontsize=8,loc='lower left')
        axes[0].set_ylabel('Time relative to SS reference (s)')
        fig.colorbar(mesh,ax=axes,label='Signed amplitude / individual SS peak',extend='both',shrink=.8)
        fig.suptitle('Paired PREM-reference timing trial · %s waveforms · %s events\n'
                     'Fixed 1° means · 10–100 s, 2 corners · PREM polarity retained for all panels'%(number,events))
        for suffix in ['png','pdf']:fig.savefig(output/('distance_sections'+('_precursors' if zoom else '')+'.'+suffix),dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    keep=(tau>=-45)&(tau<=45)
    for label,data in sections:
        numerator=np.sum(np.where(data['support']>0,data['stack'],0)*data['support'],axis=1)
        denominator=np.sum(data['support'],axis=1)
        values=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        ax.plot(tau[keep],values[keep],label=label,lw=1.7)
    ax.axvline(0,color='.7',ls=':');ax.axhline(0,color='.7',lw=.7)
    ax.set(xlabel='Time relative to SS reference (s)',ylabel='Mean signed amplitude / individual SS peak',
           title='Identical waveform pairs · timing only · PREM polarity retained')
    ax.legend()
    for suffix in ['png','pdf']:fig.savefig(output/('SS_before_after.'+suffix),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prem-plan',type=Path,required=True)
    parser.add_argument('--glad-plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-lag',type=float,default=30)
    parser.add_argument('--minimum-correlation',type=float,default=.7)
    parser.add_argument('--minimum-margin',type=float,default=.05)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists')
    if (not np.isfinite([args.max_lag,args.minimum_correlation,args.minimum_margin]).all()
            or not 0<args.max_lag<=60 or not 0<args.minimum_correlation<=1 or not 0<=args.minimum_margin<=1):
        parser.error('Invalid alignment settings')
    prem=load_global(args.prem_plan,[100,160]);glad=load_global(args.glad_plan,[100,160])
    prem=match_global(prem,glad);glad=match_global(glad,prem,True)
    lookup={(r['event_id'],r['trace_id']):i for i,r in enumerate(glad[5])}
    template_time=np.arange(-40.,41.);tau=prem[3]
    prem_data=[];before_data=[];after_data=[];prem_rows=[];glad_rows=[];prem_members=[];glad_members=[];qc=[]
    for pi,pr in enumerate(prem[5]):
        gi=lookup[pr['event_id'],pr['trace_id']];pu=prem[4][pi];gu=glad[4][gi]
        fit=absolute_template_lag(tau,gu,template_time,sample_trace(tau,pu,template_time),args.max_lag)
        reason=None
        if fit['lag_at_boundary']:reason='Paired PREM correlation lag at boundary'
        elif fit['absolute_correlation']<args.minimum_correlation:reason='Paired PREM absolute correlation below threshold'
        elif fit['peak_margin'] is not None and fit['peak_margin']<args.minimum_margin:reason='Paired PREM correlation ambiguous'
        identity=dict(event_id=pr['event_id'],trace_id=pr['trace_id'])
        qc.append(dict(identity,accepted=reason is None,reason=reason,**fit))
        if reason:continue
        prem_data.append(pu);before_data.append(gu);after_data.append(sample_trace(tau,gu,tau+fit['lag_s']))
        prem_rows.append(pr);glad_rows.append(glad[5][gi])
        prem_members.append(prem[2][pi])
        glad_members.append(dict(glad[2][gi],paired_prem_alignment=fit,paired_prem_trace_id=pr['trace_id']))
    if not prem_data:raise ValueError('No pairs pass PREM-reference alignment QC')
    args.output.mkdir(parents=True)
    prem_data,before_data,after_data=map(np.asarray,[prem_data,before_data,after_data])
    trial=dict(reference='Matching event/station aligned PREM trace',template_window_s=[-40,40],
               max_lag_s=args.max_lag,minimum_absolute_correlation=args.minimum_correlation,minimum_peak_margin=args.minimum_margin,
               fitting='Maximum absolute normalized correlation; signed coefficient recorded but not applied.',
               lag_convention='GLAD_after(t)=GLAD_before(t+lag); the entire waveform receives the same shift.',
               no_additional_polarity_changes=True,normalization='Existing individual absolute SS peak normalization retained.')
    saved=[]
    for name,data,rows,members,path,plan in [
        ('PREM',prem_data,prem_rows,prem_members,args.prem_plan,prem[0]),
        ('GLAD_before',before_data,glad_rows,glad_members,args.glad_plan,glad[0]),
        ('GLAD_after',after_data,glad_rows,glad_members,args.glad_plan,glad[0])]:
        saved.append((name.replace('_',' '),save_section(args.output/name,name,tau,data,rows,members,path,plan,trial)))
    # Retain actual shifted station traces for reuse, with pair identity and fit metadata.
    np.save(args.output/'aligned_glad_waveforms.npy',after_data,allow_pickle=False)
    np.save(args.output/'tau.npy',tau,allow_pickle=False)
    dump_json(args.output/'aligned_glad_traces.json',[dict(row,previous_measured_ss_s=row['measured_ss_s'],
                                                   measured_ss_s=row['measured_ss_s']+m['paired_prem_alignment']['lag_s'],
                                                   alignment_residual_s=row['alignment_residual_s']+m['paired_prem_alignment']['lag_s'],
                                                   alignment_method='paired_aligned_PREM_absolute_correlation_timing_only',
                                                   paired_prem_alignment=m['paired_prem_alignment'])
                                                   for row,m in zip(glad_rows,glad_members)])
    dump_json(args.output/'alignment_qc.json',qc)
    accepted=[r for r in qc if r['accepted']]
    summary=dict(trial=trial,input_pairs=len(qc),accepted_pairs=len(accepted),
        events=len({r['event_id'] for r in accepted}),rejected_reasons=dict(Counter(r['reason'] for r in qc if not r['accepted'])),
        lag_percentiles_s=np.percentile([r['lag_s'] for r in accepted],[5,25,50,75,95]).tolist(),
        median_absolute_correlation_before=float(np.median([abs(r['signed_correlation_before']) for r in accepted])),
        median_absolute_correlation_after=float(np.median([abs(r['signed_correlation_after']) for r in accepted])),
        negative_correlation_pairs_before=sum(r['signed_correlation_before']<0 for r in accepted),
        negative_correlation_pairs_after=sum(r['signed_correlation_after']<0 for r in accepted),
        counts=saved[0][1]['counts'].tolist())
    ss=(tau>=-30)&(tau<=30)
    summary['SS_stack_abspeak_ranges_1degree']={label:[float(np.min(np.max(abs(s['stack'][ss]),axis=0))),
                       float(np.max(np.max(abs(s['stack'][ss]),axis=0)))] for label,s in saved}
    dump_json(args.output/'summary.json',summary)
    plot_sections(args.output,tau,saved,len(accepted),summary['events'])
    from .wiggle_section import load_section, draw
    wiggles=args.output/'wiggles_5deg'; wiggles.mkdir()
    w={name:load_section(args.output/name,5) for name in ['PREM','GLAD_before','GLAD_after']}
    for name,section in w.items():
        np.savez_compressed(wiggles/(name+'_distance_stack.npz'),tau=section['tau'],stack=section['stack'],
                            trace_counts=section['counts'],distance_edges=section['edges'],support=section['support'])
    draw([('PREM',w['PREM']),('GLAD · aligned to PREM',w['GLAD_after'])],wiggles/'PREM_GLAD_after',35,2.5)
    draw([('GLAD · before',w['GLAD_before']),('GLAD · aligned to PREM',w['GLAD_after'])],wiggles/'GLAD_before_after',35,2.5)
    print(summary)

if __name__=='__main__':main()
