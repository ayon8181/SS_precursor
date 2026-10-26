"""One fixed cap: verify PREM arrivals, then compare model filter sensitivity."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks
from scipy.signal.windows import tukey
from .asdf_io import open_asdf,select_event,associated_stream
from .preprocessing import transverse
from .stacking import sample_trace
from .pipeline import dump_json
from .distance_section import distance_stack
from .circular_stacks import cap_memberships,read_centers

BANDS=[(10,100),(10,50),(20,100),(30,100)]
PHASES=['S^400S','S^670S']
EXTRA_INTERFERENCE=['ScS^400ScS','ScS^670ScS','sS^400S','sS^670S','pSS','S^220S']


def fixed_filter(trace,origin,cfg,band,measured_ss,amplitude,polarity,tau):
    """Filter input ASDF afresh, retaining a frozen SS time, scale and sign."""
    dt=trace.stats.delta
    if band[0]<cfg['minimum_synthetic_period_s'] or 1/band[0]>=.5/dt:
        raise ValueError('Band exceeds supported waveform bandwidth')
    u=np.asarray(trace.data,dtype=float)
    if not np.isfinite(u).all() or amplitude<=0 or polarity not in (-1.,1.):
        raise ValueError('Invalid waveform, amplitude or polarity')
    u=detrend(u,type=cfg['detrend'])*tukey(len(u),alpha=2*cfg['taper_fraction_per_end'])
    sos=butter(cfg['filter_order'],[1/band[1],1/band[0]],fs=1/dt,btype='bandpass',output='sos')
    pad=int(round(cfg['filter_padding_s']/dt))
    u=sosfiltfilt(sos,np.pad(u,pad,mode='reflect'))
    if pad:u=u[pad:-pad]
    time=float(trace.stats.starttime-origin.time)+np.arange(len(u))*dt
    guard=cfg['edge_guard_s'];valid=(time>=time[0]+guard)&(time<=time[-1]-guard)
    result=sample_trace(time[valid]-measured_ss,u[valid]*polarity/amplitude,tau)
    if not np.isfinite(result).all():raise ValueError('Frozen output window lacks valid support; cohort not altered')
    return result,float(trace.stats.starttime-origin.time)


def select_records(project,radius):
    trial=project/'outputs/glad_aligned_to_prem_trial_100_160'
    pm=json.loads((trial/'PREM/members.json').read_text());gm=json.loads((trial/'GLAD_after/members.json').read_text())
    plans={name:json.loads((project/'outputs'/db/'bin_plan/plan.json').read_text())
           for name,db in [('PREM','prem_event_reference'),('GLAD','glad_m35_event_reference')]}
    comparable=lambda cfg:{k:v for k,v in cfg.items() if k!='waveform_tag'}
    if comparable(plans['PREM']['configuration'])!=comparable(plans['GLAD']['configuration']):
        raise ValueError('Model preprocessing configurations differ beyond waveform tag')
    def metadata(members,plan):
        cache={};result=[]
        for m in members:
            if m['task'] not in cache:
                p=Path(plan['cache_root'])/'events'/('event_%06d'%m['task'])/'traces.json'
                cache[m['task']]=json.loads(p.read_text())['accepted']
            r=cache[m['task']][m['row_index']]
            if (r['event_id'],r['trace_id'])!=(m['event_id'],m['trace_id']):raise ValueError('Member identity changed')
            result.append(r)
        return result
    pr=metadata(pm,plans['PREM']);gr=metadata(gm,plans['GLAD'])
    glookup={(m['event_id'],m['trace_id']):(m,r) for m,r in zip(gm,gr)}
    previous=json.loads((project/'outputs/sy_10_100s_2corners/circular_caps/provenance.json').read_text())
    script=Path(previous['centers_script']);file=Path(previous['centers_file'])
    centers,spacing=read_centers(script,file)
    coords=[[c['latitude'],c['longitude']] for c in centers]
    groups=[]
    for phase in PHASES:
        bounce=[[r['arrivals'][phase]['reflection_latitude'],r['arrivals'][phase]['reflection_longitude']] for r in pr]
        groups.append(cap_memberships(coords,bounce,radius))
    common=[np.intersect1d(a,b) for a,b in zip(*groups)]
    ci=max(range(len(centers)),key=lambda i:len(common[i]))
    records=[]
    for i in common[ci]:
        m,r=pm[i],pr[i];other,otherrow=glookup[m['event_id'],m['trace_id']]
        if m['applied_polarity_multiplier']!=other['applied_polarity_multiplier']:raise ValueError('Unmatched polarity')
        records.append(dict(PREM=dict(member=m,row=r),GLAD=dict(member=other,row=otherrow)))
    selection=dict(center=centers[ci],radius_km=radius,count=len(records),
        events=len({r['PREM']['row']['event_id'] for r in records}),
        selection_rule='Largest common-pair count with BOTH PREM S400S/S670S bounce points within the cap; no amplitude selection.',
        centers_script=str(script),centers_file=str(file),center_spacing_km=spacing,
        center_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),records=records)
    return selection,plans


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def experiment(project,output,radius):
    if output.exists():raise FileExistsError(output)
    selection,plans=select_records(project,radius);records=selection['records'];n=len(records)
    tau=np.arange(-500.,301.);offsets=np.arange(-60.,61.);models=['PREM','GLAD']
    arrays=np.empty((2,len(BANDS),n,len(tau)))
    frozen=[];sources={};groups=defaultdict(list)
    for i,record in enumerate(records):
        for mi,model in enumerate(models):
            item=record[model];row=item['row'];member=item['member']
            lag=float(member.get('paired_prem_shift_applied_s',0.))
            # GLAD_after members explicitly record the most recent PREM-reference trial shift.
            if model=='GLAD' and 'paired_prem_shift_applied_s' not in member:
                lag=float(member['paired_prem_alignment']['lag_s'])
            ss=row['measured_ss_s']+lag
            frozen.append(dict(model=model,index=i,event_id=row['event_id'],trace_id=row['trace_id'],
                source=row['source'],frozen_measured_ss_s=ss,paired_trial_shift_s=lag,
                frozen_SS_amplitude_mm=row['ss_amplitude_mm'],polarity_multiplier=member['applied_polarity_multiplier']))
            groups[(mi,row['source'],row['event_id'])].append((i,item,ss))
    for group_number,((mi,source,event_id),items) in enumerate(groups.items()):
        model=models[mi];cfg=plans[model]['configuration'];source=Path(source)
        stat=source.stat()
        source_task=items[0][1]['member']['task']
        source_prov=json.loads((Path(plans[model]['cache_root'])/'events'/('event_%06d'%source_task)/'provenance.json').read_text())
        if (stat.st_size,stat.st_mtime_ns)!=(source_prov['source_size_bytes'],source_prov['source_mtime_ns']):
            raise ValueError('Input ASDF changed since original alignment and normalization')
        sources[str(source)]=dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        with open_asdf(source) as ds:
            event,origin=select_event(ds,event_id,cfg['origin'])
            for i,item,ss in items:
                row=item['row'];member=item['member'];group=ds.waveforms[row['station']]
                stream=associated_stream(group,cfg['waveform_tag'],event_id,len(ds.events))
                trace=transverse(stream,row['backazimuth_degree'],cfg['horizontal_convention'])
                if trace.id!=row['trace_id']:raise ValueError('ASDF trace identity mismatch')
                for bi,band in enumerate(BANDS):
                    arrays[mi,bi,i],start=fixed_filter(trace,origin,cfg,band,ss,row['ss_amplitude_mm'],
                                                       member['applied_polarity_multiplier'],tau)
                if start!=row['trace_start_from_origin_s']:raise ValueError('Input timing changed')
        print('Processed source group %s/%s: %s (%s traces)'%(group_number+1,len(groups),model,len(items)),flush=True)
    output.mkdir(parents=True)
    dump_json(output/'selection.json',selection);dump_json(output/'frozen_processing.json',frozen)
    np.save(output/'filtered_waveforms.npy',arrays,allow_pickle=False)
    np.save(output/'tau.npy',tau,allow_pickle=False)
    from obspy.taup import TauPyModel
    ray_model=TauPyModel('prem')
    discontinuities=ray_model.model.s_mod.v_mod.get_discontinuity_depths()
    if not all(np.any(np.isclose(discontinuities,d)) for d in [220,400,670]):
        raise ValueError('Unexpected PREM interface catalogue')
    catalogue_names=list(dict.fromkeys(plans['PREM']['configuration']['interfering_phases']+EXTRA_INTERFERENCE))
    expanded_catalogue=[]
    for record in records:
        row=record['PREM']['row']
        depth=float(row['source_depth_km']) if 'source_depth_km' in row else None
        if depth is None:
            task=record['PREM']['member']['task']
            cache=Path(plans['PREM']['cache_root'])/'events'/('event_%06d'%task)/'provenance.json'
            depth=json.loads(cache.read_text())['origin_depth_km']
        arrivals=ray_model.get_travel_times(depth,row['distance_degree'],phase_list=catalogue_names)
        expanded_catalogue.append([dict(phase=a.name,time_s=float(a.time),ray_parameter_s_degree=float(a.ray_param_sec_degree)) for a in arrivals])
    dump_json(output/'expanded_ray_catalogue.json',expanded_catalogue)
    phase_data=np.empty((2,len(BANDS),2,n,len(offsets)))
    predictions={};interference=[]
    for pi,phase in enumerate(PHASES):
        relative=np.array([r['PREM']['row']['arrivals'][phase]['time_s']-r['PREM']['row']['predicted_ss_s'] for r in records])
        predictions[phase]=relative
        for mi in range(2):
            for bi in range(len(BANDS)):
                phase_data[mi,bi,pi]=[sample_trace(tau,u,p+offsets) for u,p in zip(arrays[mi,bi],relative)]
        for i,record in enumerate(records):
            row=record['PREM']['row']
            for arrival in expanded_catalogue[i]:
                interference.append(dict(index=i,event_id=row['event_id'],trace_id=row['trace_id'],target=phase,
                    interfering_phase=arrival['phase'],offset_from_target_s=arrival['time_s']-row['arrivals'][phase]['time_s']))
    means=phase_data.mean(axis=3)
    semblance=np.empty_like(means)
    from scipy.ndimage import uniform_filter1d
    for mi in range(2):
        for bi,band in enumerate(BANDS):
            for pi in range(2):
                values=phase_data[mi,bi,pi]
                # Sum numerator and denominator over the same finite window BEFORE division.
                width=int(round(np.sqrt(band[0]*band[1])));width+=1-width%2
                numerator=uniform_filter1d(values.sum(axis=0)**2,size=width,mode='constant')*width
                denominator=uniform_filter1d(n*np.sum(values**2,axis=0),size=width,mode='constant')*width
                mask=denominator>np.max(denominator)*1e-10
                semblance[mi,bi,pi]=np.divide(numerator,denominator,out=np.full(len(offsets),np.nan),where=mask)
    metrics=[];peaks=[]
    for mi,model in enumerate(models):
        for bi,band in enumerate(BANDS):
            ss=arrays[mi,bi][:,(tau>=-30)&(tau<=30)]
            ss_mean=ss.mean(axis=0)
            for pi,phase in enumerate(PHASES):
                y=means[mi,bi,pi];near=abs(offsets)<=20
                k=np.flatnonzero(near)[np.argmax(y[near])]
                compare=abs(offsets)<=25
                template=np.mean([sample_trace(tau,u,offsets) for u in arrays[mi,bi]],axis=0)
                corr=float(np.corrcoef(y[compare],template[compare])[0,1])
                metrics.append(dict(model=model,band_s='%g-%g'%band,phase=phase,
                    positive_peak_offset_s=float(offsets[k]),peak_amplitude_fixed_baseline_SS=float(y[k]),
                    peak_semblance_windowed=float(semblance[mi,bi,pi,k]),
                    local_waveform_correlation_with_own_SS=corr,
                    baseline_PREM_timing_check=(bool(abs(offsets[k])<=10 and y[k]>0) if model=='PREM' and bi==0 else ''),
                    baseline_check_tolerance_s=10,mean_SS_absolute_peak_fixed_baseline=float(np.max(abs(ss_mean)))))
                candidates=find_peaks(abs(y),distance=8,prominence=.003)[0]
                for q in candidates:
                    matching=[r for r in interference if r['target']==phase and abs(r['offset_from_target_s']-offsets[q])<=10]
                    counts={name:len({r['index'] for r in matching if r['interfering_phase']==name}) for name in sorted({r['interfering_phase'] for r in matching})}
                    peaks.append(dict(model=model,band_s='%g-%g'%band,target_phase=phase,offset_s=float(offsets[q]),
                        signed_amplitude=float(y[q]),classification='Expected-positive-lobe window' if abs(offsets[q])<=10 and y[q]>0 else 'Unassigned waveform extremum; not an extra-interface detection',
                        catalogued_interference_within_10s=json.dumps(dict(counts),sort_keys=True)))
    np.savez_compressed(output/'phase_stacks.npz',models=models,bands_s=BANDS,phases=PHASES,offsets_s=offsets,
        stack=means,windowed_semblance=semblance,trace_counts=np.full((2,len(BANDS),2),n),
        **{p.replace('^','')+'_predicted_relative_s':v for p,v in predictions.items()})
    csv_write(output/'recovery_metrics.csv',metrics);csv_write(output/'candidate_peaks.csv',peaks)
    csv_write(output/'predicted_interference.csv',interference)
    recovery=[r for r in metrics if r['model']=='PREM' and r['band_s']=='10-100']
    provenance=dict(center=selection['center'],radius_km=radius,records=n,events=selection['events'],bands_s=BANDS,
        filter_order=2,filter_implementation='SciPy Butterworth bandpass, forward-reverse zero phase',
        input='Original input ASDF proc_prem/proc_glad tags, not the previously 10-100 s filtered cache arrays.',
        frozen_normalization='Each model retains its own 10-100 s individual SS amplitude for EVERY band; no per-band or stack-peak renormalization.',
        frozen_timing='Existing event-local PREM SS time and most recent paired-PREM trial GLAD time retained for EVERY band; no band-specific shifts.',
        frozen_polarity='Matching PREM preprocessing multiplier used in both models and EVERY band; no additional flips.',
        alignment_limitation='The GLAD paired-trial shifts can match different pulse lobes; they are not validated physical travel-time corrections.',
        membership='Identical records for both models, both phases and all bands; both PREM precursor bounce points inside one common cap.',
        phase_moveout='Sample each record at its individual PREM T_precursor minus T_SS plus offset; source depth handled per record.',
        PREM_baseline_recovery=recovery,baseline_timing_tolerance_s=10,
        prediction_catalogue=catalogue_names,verified_PREM_discontinuities_km=discontinuities.tolist(),
        interference_limitation='Catalogue is not exhaustive. Listed peaks are waveform extrema; no automatic artifact or extra-discontinuity classification.',
        semblance='Numerator and denominator summed over a window of sqrt(short_period*long_period) seconds before division; equal trace weights, low energy masked.',
        processing_configurations={k:p['configuration'] for k,p in plans.items()},sources=sources)
    dump_json(output/'provenance.json',provenance)
    plot_results(output,selection,tau,arrays,offsets,means,metrics,predictions,interference)
    print(json.dumps(dict(output=str(output),center=selection['center'],records=n,events=selection['events'],PREM_baseline_recovery=recovery)),flush=True)


def plot_results(output,selection,tau,arrays,offsets,means,metrics,predictions,interference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    center=selection['center'];caption='%s · %g km · N=%s, %s events · fixed records, timing, polarity and baseline SS scales'%(center['id'],selection['radius_km'],selection['count'],selection['events'])
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for pi,(phase,ax) in enumerate(zip(PHASES,axes)):
        ax.plot(offsets,means[0,0,pi],color='black',label='PREM 10–100 s')
        ax.plot(offsets,np.mean([sample_trace(tau,u,offsets) for u in arrays[0,0]],axis=0)*.045,
                color='.6',ls=':',label='SS wavelet ×0.045 (shape reference)')
        row=next(r for r in metrics if r['model']=='PREM' and r['band_s']=='10-100' and r['phase']==phase)
        ax.axvspan(-10,10,color='green',alpha=.08,label='Declared ±10 s timing tolerance')
        ax.axvline(0,color='.4',ls='--',lw=1);ax.axhline(0,color='.6',lw=.6)
        ax.plot(row['positive_peak_offset_s'],row['peak_amplitude_fixed_baseline_SS'],'o',color='green')
        ax.set(xlim=(-50,50),xlabel='Offset from individual PREM differential-time prediction (s)',
            ylabel='Amplitude / frozen 10–100 s SS peak',title='%s: positive peak %+.0f s'%(phase.replace('^',''),row['positive_peak_offset_s']))
        ax.legend(fontsize=8)
    fig.suptitle('Known PREM arrival timing check · '+caption,fontsize=11)
    save(fig,output/'PREM_recovery')
    fig,axes=plt.subplots(2,len(BANDS),figsize=(4.25*len(BANDS),8),sharex=True,sharey=True,constrained_layout=True,squeeze=False)
    limit=np.max(abs(means))*1.15
    for pi,phase in enumerate(PHASES):
        for bi,band in enumerate(BANDS):
            ax=axes[pi,bi]
            for mi,(name,color) in enumerate([('PREM','black'),('GLAD-M35','#d34329')]):
                ax.plot(offsets,means[mi,bi,pi],label=name,color=color,lw=1.3)
            ax.axvline(0,color='.5',ls='--',lw=.8);ax.axhline(0,color='.6',lw=.6)
            ax.set(xlim=(-60,60),ylim=(-limit,limit),xlabel='Offset from predicted phase (s)',title='%s · %g–%g s'%(phase.replace('^',''),*band))
            if bi==0:ax.set_ylabel('Amplitude / frozen baseline SS peak')
            ax.legend(fontsize=8)
    fig.suptitle('Filter sensitivity · '+caption+'\nPREM differential moveout applied per record; no per-band alignment or normalization',fontsize=11)
    save(fig,output/'filter_comparison')
    distances=np.array([r['PREM']['row']['distance_degree'] for r in selection['records']]);edges=np.arange(100.,161.,5.)
    fig,axes=plt.subplots(2,len(BANDS),figsize=(4.25*len(BANDS),9),sharex=True,sharey=True,constrained_layout=True,squeeze=False)
    for mi,name in enumerate(['PREM','GLAD-M35']):
        for bi,band in enumerate(BANDS):
            ax=axes[mi,bi];stack,counts,support,columns=distance_stack(tau,arrays[mi,bi],distances,np.ones(len(distances)),edges)
            mesh=ax.pcolormesh(edges,np.r_[tau-.5,tau[-1]+.5],stack,cmap='RdBu',vmin=-.08,vmax=.08,rasterized=True)
            for phase,relative in predictions.items():
                curve=[np.median(relative[columns==j]) if np.any(columns==j) else np.nan for j in range(12)]
                ax.plot((edges[:-1]+edges[1:])/2,curve,'--',lw=.8,label=phase.replace('^',''))
            ax.set(ylim=(-300,-70),xlim=(100,160),xlabel='Epicentral distance (degrees)',title='%s · %g–%g s'%(name,*band))
            if bi==0:ax.set_ylabel('Time relative to frozen SS reference (s)')
            ax.legend(fontsize=8);ax.set_facecolor('.85')
    fig.colorbar(mesh,ax=axes.ravel().tolist(),label='Signed amplitude / frozen baseline SS peak',shrink=.7,extend='both')
    fig.suptitle('Native SS-relative precursor sections · fixed 5° bins · '+caption,fontsize=11)
    save(fig,output/'distance_sections')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
    order=np.argsort(distances);rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
    names=sorted({r['interfering_phase'] for r in interference})
    for pi,(phase,ax) in enumerate(zip(PHASES,axes)):
        for j,name in enumerate(names):
            values=[r for r in interference if r['target']==phase and r['interfering_phase']==name and abs(r['offset_from_target_s'])<=100]
            if values:ax.scatter([r['offset_from_target_s'] for r in values],[rank[r['index']] for r in values],s=12,label=name,color=plt.cm.tab10(j%10))
        ax.axvline(0,color='black',ls='--',lw=.8);ax.axvspan(-10,10,color='green',alpha=.08)
        ax.set(xlim=(-100,100),xlabel='Predicted interfering arrival minus target phase (s)',title=phase.replace('^',''))
        if ax.collections:ax.legend(fontsize=8)
    axes[0].set_ylabel('Fixed records, ordered by epicentral distance')
    fig.suptitle('PREM interference catalogue (incomplete): absence of markers does not rule out interference\n'+caption,fontsize=10)
    save(fig,output/'interference_check')


def save(fig,path):
    import matplotlib.pyplot as plt
    for suffix in ['png','pdf']:fig.savefig(str(path)+'.'+suffix,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--radius-km',type=float,default=900)
    parser.add_argument('--bands',nargs='+',default=['10:100','10:50','20:100','30:100'],help='Period pairs SHORT:LONG; first must be baseline 10:100')
    args=parser.parse_args()
    global BANDS
    try:
        BANDS=[tuple(map(float,band.split(':'))) for band in args.bands]
        if (any(len(b)!=2 or not np.isfinite(b).all() or not 10<=b[0]<b[1] for b in BANDS)
                or BANDS[0]!=(10.,100.) or len(set(BANDS))!=len(BANDS)):
            raise ValueError('Invalid bands')
    except ValueError:
        parser.error('Use distinct SHORT:LONG pairs with 10 <= SHORT < LONG, starting with 10:100')
    experiment(args.project.resolve(),args.output.resolve(),args.radius_km)

if __name__=='__main__':main()
