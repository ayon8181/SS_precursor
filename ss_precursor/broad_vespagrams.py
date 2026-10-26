"""Broad linear SS-reference vespagrams with fixed paired records and bands."""
import argparse
from collections import defaultdict,Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,detrend,hilbert
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d,minimum_filter1d
from .asdf_io import open_asdf,select_event,associated_stream
from .preprocessing import transverse
from .pipeline import dump_json

BANDS=[(10,100),(10,50),(20,100),(30,100)]


def segments(mask):
    change=np.diff(np.r_[False,mask,False].astype(int))
    return zip(np.flatnonzero(change==1),np.flatnonzero(change==-1))


def broad_beam(time,data,distances,display_time,slownesses,reference,period_band,
               padding=600.,analytic_guard=300.,minimum_fraction=.8):
    """Query u_i(t + p*(D_i-D0)); p is precursor-minus-SS slowness.

    Hilbert transforms operate on expanded, supported signed stack segments,
    with reflected FFT guards. No phase-dependent curvature correction occurs.
    """
    time=np.asarray(time);data=np.asarray(data);d=np.asarray(distances)
    display_time=np.asarray(display_time);pgrid=np.asarray(slownesses)
    dt=float(display_time[1]-display_time[0]);n=len(d)
    if (data.shape!=(n,len(time)) or n<1 or not np.all(np.diff(time)>0)
            or not np.allclose(np.diff(display_time),dt) or padding<analytic_guard or analytic_guard<0):
        raise ValueError('Invalid beam inputs/padding')
    pad=int(np.ceil(padding/dt));guard=int(np.ceil(analytic_guard/dt))
    ext=display_time[0]+np.arange(-pad,len(display_time)+pad)*dt
    crop=slice(pad,pad+len(display_time));minimum=int(np.ceil(minimum_fraction*n))
    if not 1<=minimum<=n:raise ValueError('Invalid support fraction')
    window=max(1,int(round(np.sqrt(period_band[0]*period_band[1])/dt)));window+=1-window%2
    shape=(len(pgrid),len(display_time))
    signed=np.full(shape,np.nan);envelope=np.full(shape,np.nan);sem=np.full(shape,np.nan)
    counts=np.zeros(shape,dtype=np.int32);envmask=np.zeros(shape,dtype=bool)
    for ip,p in enumerate(pgrid):
        sampled=np.array([np.interp(ext+p*(di-reference),time,u,left=np.nan,right=np.nan)
                          for di,u in zip(d,data)])
        valid=np.isfinite(sampled);count=valid.sum(axis=0);values=np.where(valid,sampled,0.)
        total=values.sum(axis=0);energy=(values**2).sum(axis=0)
        stack=np.divide(total,count,out=np.full(len(ext),np.nan),where=count>0)
        eligible=(count>=minimum)&np.isfinite(stack);analytic=np.full(len(ext),np.nan)
        for lo,hi in segments(eligible):
            if hi-lo<=2*guard+2:continue
            segment=stack[lo:hi]
            reflected=np.pad(segment,guard,mode='reflect') if guard else segment
            env=np.abs(hilbert(reflected))
            if guard:env=env[guard:-guard]
            # Reflection is only an FFT boundary device, never counted as data.
            analytic[lo+guard:hi-guard]=env[guard:len(env)-guard] if guard else env
        num=uniform_filter1d(total**2,size=window,mode='constant',cval=0.)*window
        den=uniform_filter1d(count*energy,size=window,mode='constant',cval=0.)*window
        enough=minimum_filter1d(count,size=window,mode='constant',cval=0)>=minimum
        threshold=float(np.max(den))*1e-10
        coherence=np.divide(num,den,out=np.full(len(ext),np.nan),where=enough&(den>threshold)&(den>0))
        signed[ip]=stack[crop];envelope[ip]=analytic[crop];sem[ip]=np.clip(coherence[crop],0,1)
        counts[ip]=count[crop];envmask[ip]=np.isfinite(analytic[crop])
    return dict(signed_stack=signed,envelope=envelope,trace_support=counts,
                envelope_valid=envmask,windowed_semblance=sem,
                semblance_valid=np.isfinite(sem),semblance_window_samples=np.array(window),
                minimum_support=np.array(minimum),padding_s=np.array(padding),analytic_guard_s=np.array(analytic_guard))


def filter_native(trace,origin,cfg,band,ss,amplitude,sign,query):
    dt=trace.stats.delta
    if band[0]<cfg['minimum_synthetic_period_s'] or 1/band[0]>=.5/dt:
        raise ValueError('Unsupported filter band')
    u=np.asarray(trace.data,dtype=float)
    if not np.isfinite(u).all() or amplitude<=0 or sign not in (-1.,1.):raise ValueError('Invalid input')
    u=detrend(u,type=cfg['detrend'])*tukey(len(u),alpha=2*cfg['taper_fraction_per_end'])
    sos=butter(cfg['filter_order'],[1/band[1],1/band[0]],fs=1/dt,btype='bandpass',output='sos')
    pad=int(round(cfg['filter_padding_s']/dt));u=sosfiltfilt(sos,np.pad(u,pad,mode='reflect'))
    if pad:u=u[pad:-pad]
    times=float(trace.stats.starttime-origin.time)+np.arange(len(u))*dt
    valid=(times>=times[0]+cfg['edge_guard_s'])&(times<=times[-1]-cfg['edge_guard_s'])
    return np.interp(query+ss,times[valid],u[valid]*sign/amplitude,left=np.nan,right=np.nan)


def prediction_points(selection,catalogue,reference,time_limits):
    points=[]
    for i,record in enumerate(selection['records']):
        row=record['PREM']['row'];d=row['distance_degree'];ss=row['arrivals']['SS'];names=[]
        for name in ['S^400S','S^670S']:
            a=row['arrivals'][name];names.append((name,a['time_s'],a['ray_parameter_s_degree'],'precursor'))
        for a in catalogue[i]:
            relative=a['time_s']-ss['time_s']
            if time_limits[0]<=relative<=time_limits[1]:
                names.append((a['phase'],a['time_s'],a['ray_parameter_s_degree'],'other PREM phase'))
        for name,t,q,kind in names:
            p=q-ss['ray_parameter_s_degree'];relative=t-ss['time_s']
            points.append(dict(index=i,event_id=row['event_id'],trace_id=row['trace_id'],phase=name,kind=kind,
                distance_degree=d,relative_slowness_s_degree=p,time_at_record_distance_s=relative,
                projected_time_at_reference_s=relative-p*(d-reference)))
    return points


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def experiment(selection_directory,output,reference=None,pstep=.05,base_padding=900.,check_padding=1200.):
    if output.exists():raise FileExistsError(output)
    selection_directory=selection_directory.resolve()
    selection=json.loads((selection_directory/'selection.json').read_text())
    previous=json.loads((selection_directory/'provenance.json').read_text())
    catalogue=json.loads((selection_directory/'expanded_ray_catalogue.json').read_text())
    records=selection['records'];n=len(records);models=['PREM','GLAD']
    keys=[(r['PREM']['row']['event_id'],r['PREM']['row']['trace_id']) for r in records]
    if n!=127 or len(set(keys))!=n:raise ValueError('Requires the fixed 127-record S545 selection')
    distances=np.array([r['PREM']['row']['distance_degree'] for r in records])
    if reference is None:reference=float(distances.mean())
    if not distances.min()<=reference<=distances.max():raise ValueError('Reference outside aperture')
    points=prediction_points(selection,catalogue,reference,(-350,50))
    low=min(-3.,np.floor((min(r['relative_slowness_s_degree'] for r in points)-.25)/.25)*.25)
    high=max(3.,np.ceil((max(r['relative_slowness_s_degree'] for r in points)+.25)/.25)*.25)
    tmin=min(-350.,np.floor((min(r['projected_time_at_reference_s'] for r in points)-25)/25)*25)
    tmax=max(50.,np.ceil((max(r['projected_time_at_reference_s'] for r in points)+25)/25)*25)
    display=np.arange(tmin,tmax+1.);slownesses=low+np.arange(int(round((high-low)/pstep))+1)*pstep
    shift_padding=max(abs(low),abs(high))*float(np.max(abs(distances-reference)))
    query=np.arange(np.floor(tmin-check_padding-shift_padding-2),np.ceil(tmax+check_padding+shift_padding+2)+1)
    inputs=np.full((2,len(BANDS),n,len(query)),np.nan);groups=defaultdict(list);frozen=[];depths=np.full(n,np.nan)
    for i,record in enumerate(records):
        for mi,model in enumerate(models):
            row=record[model]['row'];member=record[model]['member']
            if (row['event_id'],row['trace_id'])!=keys[i]:raise ValueError('Model identities differ')
            # Ignore all paired-PREM trial shifts. Use each model's own baseline SS time.
            ss=row['measured_ss_s'];sign=member['applied_polarity_multiplier']
            if sign!=record['PREM']['member']['applied_polarity_multiplier']:raise ValueError('Polarity mismatch')
            frozen.append(dict(index=i,model=model,event_id=row['event_id'],trace_id=row['trace_id'],
                baseline_measured_SS_s=ss,normalizing_baseline_SS_amplitude_mm=row['ss_amplitude_mm'],
                polarity_multiplier=sign,ignored_paired_PREM_trial_shift_s=member.get('paired_prem_shift_applied_s',0.),
                trace_start_from_origin_s=row['trace_start_from_origin_s']))
            groups[(mi,row['source'],row['event_id'])].append((i,row,ss,sign))
    for number,((mi,path,event_id),items) in enumerate(groups.items()):
        model=models[mi];cfg=previous['processing_configurations'][model];source=Path(path);stat=source.stat()
        expected=previous['sources'][str(source)]
        if (stat.st_size,stat.st_mtime_ns)!=(expected['size_bytes'],expected['mtime_ns']):raise ValueError('Source ASDF changed')
        with open_asdf(source) as ds:
            _,origin=select_event(ds,event_id,cfg['origin'])
            for i,row,ss,sign in items:
                if mi==0:depths[i]=origin.depth/1000
                elif not np.isclose(depths[i],origin.depth/1000,atol=1e-5):raise ValueError('Model source depths differ')
                stream=associated_stream(ds.waveforms[row['station']],cfg['waveform_tag'],event_id,len(ds.events))
                trace=transverse(stream,row['backazimuth_degree'],cfg['horizontal_convention'])
                if trace.id!=row['trace_id'] or float(trace.stats.starttime-origin.time)!=row['trace_start_from_origin_s']:
                    raise ValueError('Input identity or timing changed')
                for bi,band in enumerate(BANDS):
                    inputs[mi,bi,i]=filter_native(trace,origin,cfg,band,ss,row['ss_amplitude_mm'],sign,query)
        print('Filtered source %s/%s'%(number+1,len(groups)),flush=True)
    for point in points:point['source_depth_km']=float(depths[point['index']])
    for row in frozen:row['source_depth_km']=float(depths[row['index']])
    output.mkdir(parents=True)
    dump_json(output/'selection.json',selection);dump_json(output/'frozen_processing.json',frozen)
    dump_json(output/'predictions.json',points);write_csv(output/'predictions.csv',points)
    np.save(output/'input_time_s.npy',query,allow_pickle=False)
    np.save(output/'padded_filtered_waveforms.npy',inputs,allow_pickle=False)
    panels=[];convergence=[]
    for mi,model in enumerate(models):
        for bi,band in enumerate(BANDS):
            print('Computing %s %s-%s s'%(model,*band),flush=True)
            a=broad_beam(query,inputs[mi,bi],distances,display,slownesses,reference,band,padding=base_padding)
            b=broad_beam(query,inputs[mi,bi],distances,display,slownesses,reference,band,padding=check_padding)
            joint=a['envelope_valid']&b['envelope_valid'];diff=np.abs(a['envelope']-b['envelope'])
            precursor=joint&(display[None,:]<-70)
            c=dict(model=model,band_s=list(band),first_padding_s=base_padding,second_padding_s=check_padding,
                   jointly_valid_fraction=float(joint.mean()),
                   maximum_envelope_difference_SS_units=float(np.max(diff[joint])) if joint.any() else None,
                   precursor_maximum_envelope_difference_SS_units=float(np.max(diff[precursor])) if precursor.any() else None,
                   p99_envelope_difference_SS_units=float(np.percentile(diff[joint],99)) if joint.any() else None)
            # Use the larger padding, retain the comparison; no display smoothing or rescaling.
            np.savez_compressed(output/('%s_%s_%ss.npz'%(model,*band)),time_s=display,
                relative_slowness_s_degree=slownesses,reference_distance_degree=reference,**b)
            panels.append((model,band,b));convergence.append(c)
    write_csv(output/'padding_convergence.csv',convergence)
    robust=float(np.diff(np.percentile(distances,[10,90]))[0]);aperture=float(np.ptp(distances))
    resolution=[]
    for band in BANDS:
        nominal=float(np.sqrt(band[0]*band[1])/robust)
        resolution.append(dict(band_s=list(band),nominal_period_s=float(np.sqrt(band[0]*band[1])),
            nominal_slowness_scale_s_degree=nominal,
            poor_distance_aperture=bool(robust<10),
            S400S_SS_slowness_separation_unresolved=bool(np.median([abs(r['relative_slowness_s_degree']) for r in points if r['phase']=='S^400S'])<nominal),
            S670S_SS_slowness_separation_unresolved=bool(np.median([abs(r['relative_slowness_s_degree']) for r in points if r['phase']=='S^670S'])<nominal)))
    metadata=dict(selection_directory=str(selection_directory),selection_sha256=hashlib.sha256((selection_directory/'selection.json').read_bytes()).hexdigest(),
        models=models,bands_s=BANDS,record_count=n,event_count=len({k[0] for k in keys}),center=selection['center'],radius_km=selection['radius_km'],
        reference_distance_degree=reference,reference_definition='Equal-trace mean epicentral distance of the fixed records, shared by both models and all bands.',
        beam_convention='V(t,p) = sum(valid u_i[t+p*(D_i-D0)]) / N_valid. Positive p means later relative arrivals at greater distance; t is the intercept at D0.',
        time_display_s=[tmin,tmax],initial_time_request_s=[-350,50],
        initial_slowness_request_s_degree=[-3,3],slowness_used_s_degree=[low,high,pstep],
        range_reason='Expanded to contain PREM local-tangent predictions for precursors and catalogued other phases whose record-distance differential times fall within -350 to +50 s.',
        predictions='p_i=q_phase_i-q_SS_i; t_ref_i=(T_phase_i-T_SS_i)-p_i*(D_i-D0). Points are local tangent distributions, not exact beam maxima; curvature/source-depth differences remain in the data.',
        GLAD_prediction_limitation='All overlaid predictions are PREM 1D reference rays, not GLAD-M35 3D travel-time predictions.',
        alignment='Each model own baseline measured SS time, frozen across bands. Additional paired-PREM trial shifts explicitly ignored.',
        normalization='Each model own individual 10-100 s baseline absolute SS peak, frozen across all bands; no panel or slowness normalization.',
        polarity='Existing matching PREM multipliers retained in both models; no additional flips.',
        filtering='Original ASDF inputs, 2-corner Butterworth forward-reverse bandpass; same detrend/taper/physical edge guard as baseline.',
        time_sampling_s=1,input_time_s=[float(query[0]),float(query[-1])],shift_padding_s=shift_padding,
        hilbert='Envelope of the signed stack. Supported contiguous expanded segments receive 300 s reflected FFT padding; the outer 300 s of each supported segment remain masked. No zero filling of missing observations.',
        padding_used_s=check_padding,padding_comparison=convergence,
        support='Selected records fixed. Input samples outside physical support are NaN. Means divide by valid trace counts. Envelope requires >=80% traces plus a 300 s supported analytic guard; masks saved separately.',
        semblance='Sum_time (sum_valid u)^2 / sum_time (N_valid * sum_valid u^2). Same traces in numerator and denominator at each sample; geometric-center-period window; energy below 1e-10 of row max is masked, no stabilizing epsilon.',
        distance_range_degree=[float(distances.min()),float(distances.max())],distance_aperture_degree=aperture,
        robust_distance_aperture_P10_P90_degree=robust,distance_standard_deviation_degree=float(distances.std()),source_depth_range_km=[float(depths.min()),float(depths.max())],
        nominal_resolution=resolution,resolution_note='sqrt(short_period*long_period)/(P90-P10 distance) is a diagnostic scale, not an uncertainty or grid resolution. Narrow phase clouds do not imply resolvable slopes.',
        common_envelope_limit=float(np.ceil(max(np.nanmax(r[2]['envelope']) for r in panels)*10)/10),
        precursor_envelope_limit=.08,
        processing_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata['model_display_names']={'PREM':'PREM','GLAD':'GLAD-M35'}
    metadata['prediction_distribution_note']='All 127 per-precursor local-tangent points are shown; their compact full-scale extents are expanded in the separate depth-colored prediction plot.'
    dump_json(output/'provenance.json',metadata)
    table=[]
    for row in resolution:
        item=dict(row);item['band_s']='%s-%s'%tuple(item['band_s']);item['scan_step_s_degree']=pstep
        item['interpretation']='Nominal period/aperture scale, not uncertainty; precursor-versus-SS slope separation is limited.'
        table.append(item)
    write_csv(output/'slowness_resolution.csv',table)
    plot_results(output,panels,display,slownesses,distances,points,records,metadata)
    print(json.dumps(dict(output=str(output),records=n,events=metadata['event_count'],time=metadata['time_display_s'],p=metadata['slowness_used_s_degree'],padding=convergence)),flush=True)


def plot_results(output,panels,time,p,d,points,records,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors={'S^400S':'#df6b00','S^670S':'#00a37a'}
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#b8b8b8')
    n=meta['record_count'];events=meta['event_count'];ref=meta['reference_distance_degree']
    for quantity,limit,filename in [('envelope',meta['common_envelope_limit'],'envelopes_common_scale'),
                                    ('envelope',.08,'envelopes_precursor_scale'),('windowed_semblance',1.,'windowed_semblance')]:
        fig,axes=plt.subplots(2,4,figsize=(18,10),sharex=True,sharey=True,constrained_layout=True)
        for index,(model,band,result) in enumerate(panels):
            ax=axes[index//4,index%4]
            mask=~result['envelope_valid'] if quantity=='envelope' else ~result['semblance_valid']
            values=np.ma.array(result[quantity],mask=mask)
            image=ax.pcolormesh(time,p,values,shading='auto',cmap=cmap,vmin=0,vmax=limit,rasterized=True)
            for phase,color in colors.items():
                pts=[r for r in points if r['phase']==phase]
                ax.scatter([r['projected_time_at_reference_s'] for r in pts],[r['relative_slowness_s_degree'] for r in pts],
                           s=8,facecolors='none',edgecolors=color,linewidths=.5,alpha=.35,rasterized=True)
            ax.axvline(0,color='.4',lw=.6,ls=':');ax.axhline(0,color='.5',lw=.6,ls=':')
            scale=meta['nominal_resolution'][index%4]['nominal_slowness_scale_s_degree']
            display_model='GLAD-M35' if model=='GLAD' else model
            ax.set(title='%s · %s–%s s\nnominal slope scale ≈ %.2f s/deg'%(display_model,*band,scale),
                   xlim=(time[0],time[-1]),ylim=(p[0],p[-1]))
            if index//4==1:ax.set_xlabel('Time relative to SS at D₀ (s)')
            if index%4==0:ax.set_ylabel('Slowness relative to SS (s/degree)')
        label='Windowed semblance' if quantity!='envelope' else 'Stack envelope / frozen individual baseline SS peak'
        fig.colorbar(image,ax=axes.ravel().tolist(),label=label,extend='max' if quantity=='envelope' else 'neither',shrink=.8)
        heading='Windowed semblance' if quantity!='envelope' else ('Broad stack envelopes · common scale' if limit!=.08 else 'Broad stack envelopes · precursor scale (SS saturates)')
        fig.suptitle('%s\nS545, 900 km · N=%s, E=%s · D₀=%.3f° · own baseline SS alignment · no curvature correction\n'
                     'PREM prediction clouds: orange S400S, green S670S (both rows) · gray=masked, white=zero\n'
                     'Nominal slope resolution is broader than precursor–SS slope separation; scan spacing is not resolution'
                     %(heading,n,events,ref),fontsize=11)
        save(fig,output/filename)
    fig,axes=plt.subplots(1,4,figsize=(20,4.5),constrained_layout=True)
    axes[0].hist(d,bins=np.arange(100,161,5),color='.35',edgecolor='white')
    axes[0].axvline(ref,color='red',ls='--',label='D₀=%.3f°'%ref);axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Epicentral distance (degrees)',ylabel='Records',title='127 records · 28 earthquakes')
    counts=Counter(r['PREM']['row']['event_id'] for r in records)
    axes[1].hist([r['source_depth_km'] for r in points if r['phase']=='S^400S'],bins=12,color='.35',edgecolor='white')
    axes[1].set(xlabel='Source depth (km)',ylabel='Records',title='Source-depth distribution')
    axes[2].bar(np.arange(len(counts)),sorted(counts.values()),color='.35')
    axes[2].set(xlabel='Earthquakes (ordered by record count)',ylabel='Records',title='Event representation')
    names=sorted({r['phase'] for r in points})
    for j,name in enumerate(names):
        pts=[r for r in points if r['phase']==name]
        axes[3].scatter([r['projected_time_at_reference_s'] for r in pts],[r['relative_slowness_s_degree'] for r in pts],
                        s=10,alpha=.5,label=name,color=plt.cm.tab10(j%10),rasterized=True)
    axes[3].set(xlim=(time[0],time[-1]),ylim=(p[0],p[-1]),xlabel='Projected time at D₀ (s)',
                ylabel='Relative slowness (s/degree)',title='PREM local-tangent distributions')
    axes[3].legend(fontsize=7,loc='lower right')
    fig.suptitle('Full aperture %.2f°; P10–P90 aperture %.2f° · distribution points are predictions, not picked arrivals'
                 %(meta['distance_aperture_degree'],meta['robust_distance_aperture_P10_P90_degree']),fontsize=11)
    save(fig,output/'sampling_and_predictions')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for phase,ax,color in zip(['S^400S','S^670S'],axes,colors.values()):
        pts=[r for r in points if r['phase']==phase]
        im=ax.scatter([r['projected_time_at_reference_s'] for r in pts],[r['relative_slowness_s_degree'] for r in pts],c=[r['source_depth_km'] for r in pts],s=15,cmap='viridis')
        ax.set(xlabel='Local tangent time at D₀ (s)',ylabel='Relative slowness (s/degree)',title=phase.replace('^',''))
        fig.colorbar(im,ax=ax,label='Source depth (km)')
    fig.suptitle('Per-record PREM predictions · common D₀=%.3f° · local tangent approximation; not exact beam markers'%ref)
    save(fig,output/'precursor_prediction_distributions')


def save(fig,path):
    import matplotlib.pyplot as plt
    for suffix in ['png','pdf']:fig.savefig(str(path)+'.'+suffix,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-distance',type=float)
    parser.add_argument('--slowness-step',type=float,default=.05)
    parser.add_argument('--padding',type=float,default=900)
    parser.add_argument('--check-padding',type=float,default=1200)
    args=parser.parse_args()
    if not (np.isfinite([args.slowness_step,args.padding,args.check_padding]).all()
            and args.slowness_step>0 and args.padding>=300 and args.check_padding>args.padding):
        parser.error('Require positive slowness step, padding>=300 and check-padding>padding')
    experiment(args.selection,args.output,args.reference_distance,args.slowness_step,args.padding,args.check_padding)

if __name__=='__main__':main()
