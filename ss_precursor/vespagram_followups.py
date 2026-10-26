"""Fixed-record zoom, geometry response and phase-coherence diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks
from scipy.ndimage import maximum_filter
from .broad_vespagrams import broad_beam,BANDS,write_csv
from .paired_prem_alignment import absolute_template_lag
from .event_database import template_lag
from .pipeline import dump_json

PHASES=['SS','S^400S','S^670S']
MODELS=['PREM','GLAD']


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save(fig,path):
    plt=plotting()
    for suffix in ['png','pdf']:fig.savefig(str(path)+'.'+suffix,dpi=180)
    plt.close(fig)


def load_context(broad):
    meta=json.loads((broad/'provenance.json').read_text())
    sel=json.loads((broad/'selection.json').read_text());rows=sel['records']
    frozen=json.loads((broad/'frozen_processing.json').read_text())
    depths=np.array([r['source_depth_km'] for r in frozen if r['model']=='PREM'])
    d=np.array([r['PREM']['row']['distance_degree'] for r in rows])
    inputs=np.load(broad/'padded_filtered_waveforms.npy',mmap_mode='r')
    time=np.load(broad/'input_time_s.npy')
    predictions=json.loads((broad/'predictions.json').read_text())
    return meta,sel,depths,d,inputs,time,predictions


def zoom(broad,output,meta,predictions):
    plt=plotting();fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
    arrays={};cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#b8b8b8')
    for mi,model in enumerate(MODELS):
        for bi,band in enumerate(BANDS):
            with np.load(broad/('%s_%s_%ss.npz'%(model,*band))) as a:
                ti=(a['time_s']>=-300)&(a['time_s']<=-100)
                pi=(a['relative_slowness_s_degree']>=-1.5-1e-9)&(a['relative_slowness_s_degree']<=.5+1e-9)
                time=a['time_s'][ti];slow=a['relative_slowness_s_degree'][pi]
                values=a['envelope'][np.ix_(pi,ti)];mask=a['envelope_valid'][np.ix_(pi,ti)]
                arrays['%s_%s_%ss_envelope'%(model,*band)]=values
                arrays['%s_%s_%ss_envelope_valid'%(model,*band)]=mask
            ax=axes[mi,bi]
            im=ax.pcolormesh(time,slow,np.ma.array(values,mask=~mask),cmap=cmap,vmin=0,vmax=.08,shading='auto',rasterized=True)
            for phase,color in [('S^400S','#df6b00'),('S^670S','#00a37a')]:
                pp=[r for r in predictions if r['phase']==phase]
                ax.scatter([r['projected_time_at_reference_s'] for r in pp],[r['relative_slowness_s_degree'] for r in pp],
                           facecolors='none',edgecolors=color,s=4,linewidths=.25,alpha=.18,rasterized=True)
            ax.set(xlim=(-300,-100),ylim=(-1.5,.5),title='%s · %s–%s s'%('GLAD-M35' if model=='GLAD' else model,*band))
            if mi==1:ax.set_xlabel('Time relative to baseline SS at D₀ (s)')
            if bi==0:ax.set_ylabel('Slowness relative to SS (s/degree)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Envelope / frozen baseline SS peak',extend='max',shrink=.8)
    fig.suptitle('Zoom of EXISTING arrays · common precursor scale 0–0.08 · no processing changes\n'
                 'S545, 900 km · 127 records, 28 events · small hollow PREM prediction clouds · gray=masked',fontsize=12)
    save(fig,output/'existing_arrays_zoom')
    np.savez_compressed(output/'existing_arrays_zoom.npz',time_s=time,relative_slowness_s_degree=slow,**arrays)


def half_width(x,y,k):
    half=y[k]/2;left=k;right=k
    while left>0 and y[left-1]>=half:left-=1
    while right<len(y)-1 and y[right+1]>=half:right+=1
    # Linear crossing interpolation, rather than reporting the scan spacing as resolution.
    lo=x[left] if left==0 else np.interp(half,[y[left-1],y[left]],[x[left-1],x[left]])
    hi=x[right] if right==len(y)-1 else np.interp(half,[y[right+1],y[right]],[x[right+1],x[right]])
    return float(hi-lo),bool(left==0 or right==len(y)-1)


def response_checks(broad,output,meta,sel,d,inputs,time,predictions):
    plt=plotting();ref=meta['reference_distance_degree'];x=d-ref;n=len(d)
    with np.load(broad/'PREM_10_100s.npz') as a:display=a['time_s'];slow=a['relative_slowness_s_degree']
    native_dt=min(r['PREM']['row']['input_delta_s'] for r in sel['records'])
    native_time=np.arange(np.floor(time[0])-500,np.ceil(time[-1])+501,native_dt)
    impulse=np.zeros(len(native_time));impulse[np.argmin(abs(native_time))]=1.
    wavelets=[]
    for band in BANDS:
        sos=butter(2,[1/band[1],1/band[0]],fs=1/native_dt,btype='bandpass',output='sos')
        wavelets.append(sosfiltfilt(sos,impulse))
    scale=float(wavelets[0][np.argmin(abs(native_time))])
    wavelets=[w/scale for w in wavelets]
    # Shared physical-support mask, independent of phase and band.
    common_support=np.isfinite(inputs[0,0])&np.isfinite(inputs[1,0])
    metrics=[];sidelobes=[];leakage=[];panels={}
    out=output/'geometry_response';out.mkdir()
    for phase in PHASES:
        if phase=='SS':arrivals=np.zeros(n)
        else:arrivals=np.array([r['PREM']['row']['arrivals'][phase]['time_s']-r['PREM']['row']['predicted_ss_s'] for r in sel['records']])
        ls_p=float(np.dot(x,arrivals-arrivals.mean())/np.dot(x,x));ls_t=float(arrivals.mean())
        fit_resid=arrivals-(ls_t+ls_p*x)
        local=[p for p in predictions if p['phase']==phase]
        median_p=float(np.median([p['relative_slowness_s_degree'] for p in local])) if local else 0.
        for bi,band in enumerate(BANDS):
            data=np.array([np.interp(time-ti,native_time,wavelets[bi],left=0.,right=0.) for ti in arrivals])
            data=np.where(common_support,data,np.nan)
            print('Geometry response %s %s-%s s'%(phase,*band),flush=True)
            result=broad_beam(time,data,d,display,slow,ref,band,padding=meta['padding_used_s'])
            np.savez_compressed(out/('%s_%s_%ss.npz'%(phase.replace('^',''),*band)),time_s=display,
                relative_slowness_s_degree=slow,injected_arrival_times_s=arrivals,**result)
            panels[phase,bi]=result
            roi=(abs(display[None,:]-ls_t)<=40)&(abs(slow[:,None]-ls_p)<=1.5)
            score=np.where(roi,result['envelope'],-np.inf)
            ip,it=np.unravel_index(np.nanargmax(score),score.shape);peak=float(result['envelope'][ip,it])
            temporal_width,tc=half_width(display,result['envelope'][ip],it)
            curve=np.max(result['envelope'][:,abs(display-ls_t)<=40],axis=1)
            slope_width,pc=half_width(slow,curve,ip)
            metrics.append(dict(phase=phase,band_s='%s-%s'%band,records=n,
                injected_unit_baseline_peak=1.,single_record_same_band_peak=float(wavelets[bi][np.argmin(abs(native_time))]),
                recovered_fraction_of_same_band_impulse_peak=peak/float(wavelets[bi][np.argmin(abs(native_time))]),
                recovered_peak=peak,time_peak_s=float(display[it]),slowness_peak_s_degree=float(slow[ip]),
                least_squares_reference_time_s=ls_t,least_squares_slowness_s_degree=ls_p,
                median_local_tangent_slowness_s_degree=median_p,
                time_bias_from_linear_fit_s=float(display[it]-ls_t),slowness_bias_from_linear_fit=float(slow[ip]-ls_p),
                slowness_bias_from_median_local_tangent=float(slow[ip]-median_p),
                prediction_nonlinearity_RMS_s=float(np.sqrt(np.mean(fit_resid**2))),
                temporal_envelope_FWHM_s=temporal_width,slowness_profile_FWHM_s_degree=slope_width,
                temporal_FWHM_hits_boundary=tc,slowness_FWHM_hits_boundary=pc))
            localmax=(result['envelope']==maximum_filter(result['envelope'],size=(3,5),mode='nearest'))
            separate=(abs(display[None,:]-display[it])>temporal_width)|(abs(slow[:,None]-slow[ip])>slope_width)
            candidates=np.argwhere(localmax&separate&result['envelope_valid'])
            if len(candidates):
                order=np.argsort([-result['envelope'][i,j] for i,j in candidates])[:10]
                for q in order:
                    i,j=candidates[q];sidelobes.append(dict(phase=phase,band_s='%s-%s'%band,time_s=float(display[j]),
                        slowness_s_degree=float(slow[i]),relative_to_response_peak=float(result['envelope'][i,j]/peak)))
            if phase=='SS':
                for target in PHASES[1:]:
                    pp=[r for r in predictions if r['phase']==target]
                    ct=np.median([r['projected_time_at_reference_s'] for r in pp]);cp=np.median([r['relative_slowness_s_degree'] for r in pp])
                    region=(abs(display[None,:]-ct)<=30)&(abs(slow[:,None]-cp)<=.5)
                    leakage.append(dict(band_s='%s-%s'%band,target=target,
                        maximum_unit_SS_only_envelope_near_target=float(np.max(result['envelope'][region]))))
    write_csv(out/'response_metrics.csv',metrics);write_csv(out/'response_secondary_maxima.csv',sidelobes)
    write_csv(out/'SS_only_leakage_near_precursors.csv',leakage)
    for limit,label in [(1.,'unit_scale'),(.08,'precursor_scale')]:
        fig,axes=plt.subplots(3,4,figsize=(17,12),sharex=True,sharey=True,constrained_layout=True)
        for pi,phase in enumerate(PHASES):
            for bi,band in enumerate(BANDS):
                ax=axes[pi,bi];r=panels[phase,bi]
                im=ax.pcolormesh(display,slow,np.ma.array(r['envelope'],mask=~r['envelope_valid']),cmap='Blues',vmin=0,vmax=limit,shading='auto',rasterized=True)
                mm=next(m for m in metrics if m['phase']==phase and m['band_s']=='%s-%s'%band)
                ax.set(title='%s only · %s–%s s\nΔp FWHM %.2f s/deg'%(phase.replace('^',''),*band,mm['slowness_profile_FWHM_s_degree']))
                if pi==2:ax.set_xlabel('SS-reference intercept time (s)')
                if bi==0:ax.set_ylabel('Relative slowness (s/degree)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Controlled envelope / unit baseline impulse peak',extend='max',shrink=.8)
        fig.suptitle('Exact 127-record geometry · phases injected separately · native filtered impulses, not data\n'
                     'Identical positive source pulses isolate geometry/filter effects; same broad stack, masks and padding',fontsize=12)
        save(fig,out/('isolated_phase_response_'+label))
    draw_SS_response(out,[panels['SS',bi] for bi in range(4)],display,slow,predictions)
    return metrics,leakage


def draw_SS_response(out,panels,display,slow,predictions):
    plt=plotting()
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),sharex=True,sharey=True,constrained_layout=True)
    for ax,band,result in zip(axes,BANDS,panels):
        im=ax.pcolormesh(display,slow,result['envelope'],shading='auto',cmap='Blues',vmin=0,vmax=.08,rasterized=True)
        for phase,color in [('S^400S','#df6b00'),('S^670S','#00a37a')]:
            pts=[r for r in predictions if r['phase']==phase]
            ct=np.median([r['projected_time_at_reference_s'] for r in pts]);cp=np.median([r['relative_slowness_s_degree'] for r in pts])
            ax.add_patch(Rectangle((ct-30,cp-.5),60,1,fill=False,edgecolor=color,linewidth=.7))
        ax.set(title='SS only · %s–%s s'%band,xlabel='SS-reference intercept time (s)')
    axes[0].set_ylabel('Relative slowness (s/degree)')
    fig.colorbar(im,ax=axes.tolist(),label='Unit-SS-only envelope; precursor scale',extend='max',shrink=.8)
    fig.suptitle('Only SS was injected · exact 127-record geometry · no precursors present'+chr(10)+
                 'Boxes mark the S400S/S670S comparison windows, not injected arrivals',fontsize=12)
    save(fig,out/'SS_only_precursor_scale')


def coherence(values,offsets,half_window=25):
    keep=abs(offsets)<=half_window;x=np.asarray(values)[:,keep]
    valid=np.isfinite(x);n=valid.sum(0);u=np.where(valid,x,0.)
    num=np.sum(u.sum(0)**2);den=np.sum(n*np.sum(u*u,axis=0))
    return float(num/den) if den>0 else np.nan


def phase_samples(time,data,times,offsets):
    return np.array([np.interp(offsets+ti,time,u,left=np.nan,right=np.nan) for u,ti in zip(data,times)])


def timing_fit(time,trace,template_time,template):
    if np.sqrt(np.mean(template**2))<.001:
        return dict(accepted=False,reason='Low template energy',lag_s=None,signed_correlation_before=None,signed_correlation_after=None)
    search=np.asarray(trace)[abs(np.asarray(time))<=55]
    if not np.isfinite(search).all() or np.std(search)<.001:
        return dict(accepted=False,reason='Low or invalid target energy',lag_s=None,signed_correlation_before=None,signed_correlation_after=None)
    try:fit=absolute_template_lag(time,trace,template_time,template,max_lag=30.,separation=10.)
    except ValueError as error:
        return dict(accepted=False,reason=str(error),lag_s=None,signed_correlation_before=None,signed_correlation_after=None)
    reason=None
    if fit['lag_at_boundary']:reason='Lag at boundary'
    elif fit['absolute_correlation']<.7:reason='Low absolute correlation'
    elif fit['peak_margin'] is not None and fit['peak_margin']<.05:reason='Ambiguous correlation peak'
    return dict(fit,accepted=reason is None,reason=reason)


def conservative_timing_fit(time,trace,template_time,template):
    """Signed fit, +/-8 s, no polarity reversal; event-excluded templates upstream."""
    search=np.asarray(trace)[abs(np.asarray(time))<=33]
    if (not np.isfinite(template).all() or np.sqrt(np.mean(template**2))<.001
            or not np.isfinite(search).all() or np.std(search)<.001):
        return dict(accepted=False,reason='Low or invalid phase/template energy',lag_s=None,
                    signed_correlation_before=None,signed_correlation_after=None)
    try:fit=template_lag(time,trace,template_time,template,max_lag_s=8.,separation_s=5.)
    except ValueError as error:
        return dict(accepted=False,reason=str(error),lag_s=None,signed_correlation_before=None,signed_correlation_after=None)
    before=float(np.corrcoef(np.interp(template_time,time,trace),template)[0,1])
    after=float(np.corrcoef(np.interp(template_time+fit['lag_s'],time,trace),template)[0,1])
    reason=None
    if fit['lag_at_boundary']:reason='Conservative lag at boundary'
    elif fit['correlation']<.7:reason='Low signed correlation'
    elif fit['peak_margin'] is not None and fit['peak_margin']<.05:reason='Ambiguous signed-correlation peak'
    return dict(fit,accepted=reason is None,reason=reason,absolute_correlation=abs(fit['correlation']),
                signed_correlation_before=before,signed_correlation_after=after,additional_polarity_multiplier=1.)


def event_excluded_template(values,events,event,offsets,template_time):
    mask=np.asarray(events)!=event
    if not np.any(mask):return np.full(len(template_time),np.nan)
    return np.interp(template_time,offsets,np.mean(values[mask],axis=0))


def coherence_checks(output,meta,sel,depths,d,inputs,time,predictions):
    out=output/'coherence_diagnostics';out.mkdir();n=len(d);ref=meta['reference_distance_degree']
    offsets=np.arange(-100.,101.);tt=np.arange(-25.,26.)
    exact=np.empty((2,4,2,n,len(offsets)));linear=np.empty_like(exact);aligned=np.empty_like(exact)
    fits=[];wide_fits=[];wide_metrics=[];metrics=[];group_metrics=[];ls=[]
    references=np.empty((2,4,2,n,len(tt)))
    events=np.array([r['PREM']['row']['event_id'] for r in sel['records']])
    depth_masks=[(label,(depths>=lo)&(depths<hi)) for label,lo,hi in [('0–70 km',0,70),('70–300 km',70,300),('300–700 km',300,700)]]
    event_counts=Counter(events);best_events=sorted([e for e in event_counts if event_counts[e]>=4],key=lambda e:(-event_counts[e],e))[:6]
    groups=[('all',np.ones(n,dtype=bool))]+depth_masks+[(e,events==e) for e in best_events]
    for pi,phase in enumerate(PHASES[1:]):
        arrivals=np.array([r['PREM']['row']['arrivals'][phase]['time_s']-r['PREM']['row']['predicted_ss_s'] for r in sel['records']])
        pp=[r for r in predictions if r['phase']==phase]
        p0=float(np.median([r['relative_slowness_s_degree'] for r in pp]));t0=float(np.median([r['projected_time_at_reference_s'] for r in pp]))
        approx=t0+p0*(d-ref);ls.append(dict(phase=phase,linear_reference_time_s=t0,linear_slowness_s_degree=p0,
            exact_minus_linear_RMS_s=float(np.sqrt(np.mean((arrivals-approx)**2))),
            exact_minus_linear_range_s=[float(np.min(arrivals-approx)),float(np.max(arrivals-approx))]))
        for bi,band in enumerate(BANDS):
            # The common PREM SS stack supplies a fixed, high-SNR pulse-shape template.
            template=np.mean(phase_samples(time,inputs[0,bi],np.zeros(n),tt),axis=0)
            for mi,model in enumerate(MODELS):
                values=phase_samples(time,inputs[mi,bi],arrivals,offsets)
                approx_values=phase_samples(time,inputs[mi,bi],approx,offsets)
                exact[mi,bi,pi]=values;linear[mi,bi,pi]=approx_values
                corrected=values.copy();wide_corrected=values.copy();accepted=np.zeros(n,dtype=bool)
                for i,u in enumerate(values):
                    wide=timing_fit(offsets,u,tt,template)
                    wide_fits.append(dict(model=model,band_s='%s-%s'%band,phase=phase,index=i,event_id=events[i],**wide))
                    if wide['accepted']:wide_corrected[i]=np.interp(offsets+wide['lag_s'],offsets,u,left=np.nan,right=np.nan)
                    leave_out=event_excluded_template(values,events,events[i],offsets,tt)
                    references[mi,bi,pi,i]=leave_out
                    fit=conservative_timing_fit(offsets,u,tt,leave_out)
                    fit['excluded_template_event_id']=events[i]
                    fit['signed_correlation_to_PREM_SS_at_prediction']=float(np.corrcoef(np.interp(tt,offsets,u),template)[0,1])
                    if fit['accepted']:
                        accepted[i]=True
                        corrected[i]=np.interp(offsets+fit['lag_s'],offsets,u,left=np.nan,right=np.nan)
                    row=sel['records'][i]['PREM']['row']
                    fits.append(dict(model=model,band_s='%s-%s'%band,phase=phase,index=i,
                        event_id=events[i],trace_id=row['trace_id'],source_depth_km=float(depths[i]),distance_degree=float(d[i]),**fit))
                aligned[mi,bi,pi]=corrected
                wide_metrics.append(dict(model=model,band_s='%s-%s'%band,phase=phase,records=n,
                    exact_prediction_coherence=coherence(values,offsets),
                    timing_diagnostic_coherence_same_records=coherence(wide_corrected,offsets),
                    accepted_timing_fits=sum(r['accepted'] for r in wide_fits if r['model']==model and r['phase']==phase and r['band_s']=='%s-%s'%band)))
                for label,mask in groups:
                    group_metrics.append(dict(model=model,band_s='%s-%s'%band,phase=phase,group=label,
                        records=int(mask.sum()),events=len(set(events[mask])),
                        linear_coherence=coherence(approx_values[mask],offsets),
                        exact_prediction_coherence=coherence(values[mask],offsets),
                        diagnostic_accepted_fits_shifted_coherence=coherence(corrected[mask],offsets),
                        accepted_fits=int(accepted[mask].sum())))
                k=(abs(offsets)<=25)
                c0=coherence(values,offsets);c1=coherence(corrected,offsets)
                localfits=[r for r in fits if r['model']==model and r['band_s']=='%s-%s'%band and r['phase']==phase and r['accepted']]
                metrics.append(dict(model=model,band_s='%s-%s'%band,phase=phase,records=n,
                    linear_coherence=coherence(approx_values,offsets),exact_prediction_coherence=c0,
                    timing_diagnostic_coherence_same_records=c1,accepted_timing_fits=int(accepted.sum()),
                    exact_stack_abspeak=float(np.max(abs(values.mean(0)[k]))),
                    timing_diagnostic_stack_abspeak=float(np.max(abs(np.nanmean(corrected,axis=0)[k]))),
                    negative_signed_fits=sum(r['signed_correlation_after']<0 for r in localfits),
                    accepted_lag_median_s=float(np.median([r['lag_s'] for r in localfits])) if localfits else None,
                    accepted_lag_P10_s=float(np.percentile([r['lag_s'] for r in localfits],10)) if localfits else None,
                    accepted_lag_P90_s=float(np.percentile([r['lag_s'] for r in localfits],90)) if localfits else None,
                    accepted_median_absolute_correlation=float(np.median([abs(r['signed_correlation_after']) for r in localfits])) if localfits else None,
                    negative_PREM_SS_shape_correlations_at_prediction=sum(r.get('signed_correlation_to_PREM_SS_at_prediction',0)<0 for r in fits if r['model']==model and r['band_s']=='%s-%s'%band and r['phase']==phase)))
    write_csv(out/'exploratory_wide_lag_coherence_metrics.csv',wide_metrics)
    dump_json(out/'exploratory_wide_lag_SS_wavelet_fits.json',wide_fits)
    dump_json(out/'timing_fits.json',fits);write_csv(out/'coherence_metrics.csv',metrics);write_csv(out/'group_metrics.csv',group_metrics)
    dump_json(out/'linear_approximation_errors.json',ls)
    write_csv(out/'timing_residuals.csv',[{k:r.get(k) for k in ['model','band_s','phase','index','event_id','trace_id','source_depth_km','distance_degree','accepted','reason','lag_s','absolute_correlation','signed_correlation_before','signed_correlation_after','signed_correlation_to_PREM_SS_at_prediction','peak_margin']} for r in fits])
    np.savez_compressed(out/'contributing_signed_traces.npz',offsets_s=offsets,models=MODELS,bands_s=BANDS,phases=PHASES[1:],
        exact_prediction_traces=exact,linear_approximation_traces=linear,
        timing_diagnostic_traces=aligned,timing_reference_templates=references,template_offsets_s=tt,source_depth_km=depths,distance_degree=d,event_ids=events)
    variance_diagnostics(out,exact,offsets,events,depths)
    write_amplitude_leverage(out,exact,offsets,sel,depths)
    draw_coherence(out,sel,depths,offsets,exact,linear,aligned,fits,metrics,depth_masks,best_events)
    return metrics,ls


def write_amplitude_leverage(out,exact,offsets,selection,depths):
    rows=[]
    for mi,model in enumerate(MODELS):
        for bi,band in enumerate(BANDS):
            for pi,phase in enumerate(PHASES[1:]):
                values=exact[mi,bi,pi][:,abs(offsets)<=25];energy=np.sum(values**2,axis=1)
                for i in np.argsort(-energy)[:10]:
                    row=selection['records'][i][model]['row']
                    rows.append(dict(model=model,band_s='%s-%s'%band,phase=phase,index=int(i),
                        event_id=row['event_id'],trace_id=row['trace_id'],source_depth_km=float(depths[i]),
                        fraction_of_window_energy=float(energy[i]/energy.sum()),
                        normalizing_baseline_SS_amplitude_mm=row['ss_amplitude_mm'],
                        window_RMS_baseline_SS_units=float(np.sqrt(np.mean(values[i]**2)))))
    write_csv(out/'amplitude_leverage.csv',rows)


def variance_split(values,labels,offsets):
    x=np.asarray(values)[:,abs(offsets)<=25];labels=np.asarray(labels)
    if not np.isfinite(x).all():raise ValueError('Variance split needs common finite support')
    energy=float(np.sum(x*x))
    if energy<=0:return dict(global_coherent_fraction=None,between_group_fraction=None,within_group_fraction=None)
    mean=x.mean(0);coherent=len(x)*float(np.sum(mean*mean));between=within=0.
    for group in np.unique(labels):
        u=x[labels==group];m=u.mean(0)
        between+=len(u)*float(np.sum((m-mean)**2));within+=float(np.sum((u-m)**2))
    return dict(global_coherent_fraction=coherent/energy,between_group_fraction=between/energy,within_group_fraction=within/energy)


def amplitude_shape_metrics(values,offsets):
    x=np.asarray(values)[:,abs(offsets)<=25];norm=np.linalg.norm(x,axis=1)
    energy=np.sum(norm**2);n=len(x);mean_energy=np.sum(x.sum(0)**2)
    raw=mean_energy/(n*energy) if energy>0 else np.nan
    ceiling=np.sum(norm)**2/(n*energy) if energy>0 else np.nan
    supported=norm>1e-12
    shape=x[supported]/norm[supported,None]
    shape_coherence=float(np.sum(shape.sum(0)**2)/(len(shape)**2)) if len(shape) else np.nan
    return dict(raw_coherence=float(raw),amplitude_only_coherence_ceiling=float(ceiling),
                coherence_fraction_of_amplitude_ceiling=float(raw/ceiling) if ceiling>0 else np.nan,
                unit_norm_shape_coherence=shape_coherence,nonzero_shape_records=int(supported.sum()),
                RMS_P10=float(np.percentile(norm/np.sqrt(x.shape[1]),10)),
                RMS_P50=float(np.percentile(norm/np.sqrt(x.shape[1]),50)),
                RMS_P90=float(np.percentile(norm/np.sqrt(x.shape[1]),90)),
                largest_record_fraction_of_energy=float(np.max(norm**2)/energy) if energy>0 else np.nan)


def variance_diagnostics(out,exact,offsets,events,depths):
    rows=[];shape_rows=[]
    for mi,model in enumerate(MODELS):
        for bi,band in enumerate(BANDS):
            for pi,phase in enumerate(PHASES[1:]):
                shape_rows.append(dict(model=model,band_s='%s-%s'%band,phase=phase,**amplitude_shape_metrics(exact[mi,bi,pi],offsets)))
                for name,labels in [('events',events),('depth_groups',np.digitize(depths,[70,300]))]:
                    rows.append(dict(model=model,band_s='%s-%s'%band,phase=phase,groups=name,
                                     **variance_split(exact[mi,bi,pi],labels,offsets)))
    write_csv(out/'waveform_energy_decomposition.csv',rows)
    write_csv(out/'waveform_shape_amplitude_metrics.csv',shape_rows)
    plt=plotting();fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True,constrained_layout=True)
    for pi,phase in enumerate(PHASES[1:]):
        for bi,band in enumerate(BANDS):
            ax=axes[pi,bi];bottom=np.zeros(2)
            for field,label,color in [('global_coherent_fraction','Global coherent mean','#42a475'),('between_group_fraction','Between-event mean differences','#efa63d'),('within_group_fraction','Within-event differences','#497cae')]:
                heights=[next(r[field] for r in rows if r['model']==model and r['band_s']=='%s-%s'%band and r['phase']==phase and r['groups']=='events') for model in MODELS]
                ax.bar([0,1],heights,bottom=bottom,color=color,label=label);bottom+=heights
            ax.set(xticks=[0,1],xticklabels=['PREM','GLAD'],ylim=(0,1),title='%s · %s–%s s'%(phase.replace('^',''),*band))
            if bi==0:ax.set_ylabel('Fraction of signed waveform energy')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.005))
    fig.suptitle('Exact PREM moveout · central ±25 s · identical records and SS scales\n'
                 'Descriptive energy identity; group sample size/noise contributes to these fractions',fontsize=11)
    save(fig,out/'waveform_energy_decomposition')
    return rows


def draw_coherence(out,sel,depths,offsets,exact,linear,aligned,fits,metrics,depth_masks,best_events):
    plt=plotting();keep=abs(offsets)<=60;n=len(sel['records']);sort=np.argsort(depths,kind='stable');rank=np.empty(n,int);rank[sort]=np.arange(n)
    for bi,band in enumerate(BANDS):
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True,constrained_layout=True)
        for mi,model in enumerate(MODELS):
            for pi,phase in enumerate(PHASES[1:]):
                ax=axes[mi,pi]
                im=ax.pcolormesh(offsets[keep],np.arange(n),exact[mi,bi,pi,sort][:,keep],shading='auto',cmap='RdBu_r',vmin=-.12,vmax=.12,rasterized=True)
                matching=[r for r in fits if r['model']==model and r['phase']==phase and r['band_s']=='%s-%s'%band]
                for accepted,marker in [(False,'x'),(True,'o')]:
                    ff=[r for r in matching if r['accepted']==accepted and r.get('lag_s') is not None]
                    if ff:ax.scatter([r['lag_s'] for r in ff],[rank[r['index']] for r in ff],s=9,
                        c=['#7b3294' if (r.get('signed_correlation_after') or 0)<0 else '#00a650' for r in ff],
                        marker=marker,linewidths=.5,alpha=.65)
                ax.axvline(0,color='.4',ls=':',lw=.8)
                ax.set(xlim=(-60,60),ylim=(-.5,n-.5),title='%s · %s · %s–%s s'%(model,phase.replace('^',''),*band))
                if mi==1:ax.set_xlabel('Offset from individual PREM phase prediction (s)')
                if pi==0:ax.set_ylabel('Same records sorted by source depth')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Signed amplitude / frozen baseline SS peak',extend='both',shrink=.8)
        fig.suptitle('Contributing SIGNED traces · exact predicted moveout · no per-trace amplitude normalization\n'
                     'Dots: accepted signed event-excluded window fits (±8 s); crosses: failed or ambiguous fits',fontsize=11)
        save(fig,out/('contributing_traces_%s_%ss'%band))
    for diagnostic,name in [(False,'linear_vs_exact_moveout'),(True,'timing_only_alignment_diagnostic')]:
        fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
        ylim=max(.08,float(np.nanmax(abs(np.nanmean(aligned if diagnostic else exact,axis=3))))*1.1)
        for pi,phase in enumerate(PHASES[1:]):
            for bi,band in enumerate(BANDS):
                ax=axes[pi,bi]
                if diagnostic:
                    ax.plot(offsets,exact[0,bi,pi].mean(0),color='black',lw=1.1,label='PREM exact prediction')
                    ax.plot(offsets,exact[1,bi,pi].mean(0),color='#d34325',lw=1.2,label='GLAD exact prediction')
                    ax.plot(offsets,np.nanmean(aligned[1,bi,pi],axis=0),color='#2171b5',lw=1.2,label='GLAD accepted fits shifted')
                else:
                    for mi,color,model in [(0,'black','PREM'),(1,'#d34325','GLAD')]:
                        ax.plot(offsets,linear[mi,bi,pi].mean(0),'--',color=color,lw=1,label=model+' linear')
                        ax.plot(offsets,exact[mi,bi,pi].mean(0),color=color,lw=1.2,label=model+' exact prediction')
                metric=next(r for r in metrics if r['model']=='GLAD' and r['phase']==phase and r['band_s']=='%s-%s'%band)
                note=' · fits %s/127'%metric['accepted_timing_fits'] if diagnostic else ''
                ax.set(xlim=(-60,60),ylim=(-ylim,ylim),title='%s · %s–%s s%s'%(phase.replace('^',''),*band,note))
                ax.axvline(0,color='.5',ls=':',lw=.7);ax.axhline(0,color='.6',lw=.6)
                if pi==1:ax.set_xlabel('Offset from model phase (s)')
                if bi==0:ax.set_ylabel('Signed amplitude / baseline SS peak')
                ax.legend(fontsize=7)
        heading='Timing-only diagnostic: signed event-excluded fits (±8 s); failed fits unchanged' if diagnostic else 'Linear broad-moveout approximation versus exact per-record PREM differential moveout'
        fig.suptitle(heading+'\nSame 127 records, amplitudes and polarities; diagnostic corrections are NOT applied to the broad overview',fontsize=11)
        save(fig,out/name)
    fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
    for pi,phase in enumerate(PHASES[1:]):
        for bi,band in enumerate(BANDS):
            ax=axes[pi,bi]
            for model,color,dx in [('PREM','black',-2),('GLAD','#d34325',2)]:
                ff=[r for r in fits if r['model']==model and r['phase']==phase and r['band_s']=='%s-%s'%band and r['accepted']]
                for neg,marker in [(False,'o'),(True,'v')]:
                    sub=[r for r in ff if (r['signed_correlation_after']<0)==neg]
                    if sub:ax.scatter([r['source_depth_km']+dx for r in sub],[r['lag_s'] for r in sub],s=12,color=color,marker=marker,alpha=.6,label=model+(' negative fit' if neg else ' positive fit'))
            ax.axhline(0,color='.5',ls=':',lw=.7);ax.set(title='%s · %s–%s s'%(phase.replace('^',''),*band),ylim=(-9,9))
            if pi==1:ax.set_xlabel('Source depth (km); ±2 km plotting offset')
            if bi==0:ax.set_ylabel('Accepted template-fit lag (s)')
            ax.legend(fontsize=6)
    fig.suptitle('Timing residual DIAGNOSTIC · signed fits to same-model, same-phase event-excluded templates\n'
                 'Lag restricted to ±8 s; polarity retained; fitted lag is not automatically a physical phase delay',fontsize=11)
    save(fig,out/'timing_residuals_by_depth')
    events=np.array([r['PREM']['row']['event_id'] for r in sel['records']])
    for group_list,name in [(depth_masks,'fixed_depth_groups_10_100'),([(e,events==e) for e in best_events],'best_sampled_events_10_100')]:
        fig,axes=plt.subplots(2,len(group_list),figsize=(4*len(group_list),8),sharex=True,sharey=True,constrained_layout=True,squeeze=False)
        maximum=max(np.max(abs(exact[mi,0,pi,mask].mean(0)[keep])) for _,mask in group_list for mi in range(2) for pi in range(2))*1.1
        for gi,(label,mask) in enumerate(group_list):
            for pi,phase in enumerate(PHASES[1:]):
                ax=axes[pi,gi]
                for mi,color,model in [(0,'black','PREM'),(1,'#d34325','GLAD')]:ax.plot(offsets,exact[mi,0,pi,mask].mean(0),color=color,label=model)
                short=label.split('/')[-1].split('#')[0]
                ax.set(xlim=(-60,60),ylim=(-maximum,maximum),title='%s · %s\nN=%s, E=%s'%(phase.replace('^',''),short,mask.sum(),len(set(events[mask]))))
                ax.axvline(0,color='.4',ls=':',lw=.7);ax.axhline(0,color='.6',lw=.6)
                if pi==1:ax.set_xlabel('Offset from predicted phase (s)')
                if gi==0:ax.set_ylabel('Signed amplitude / baseline SS peak')
                ax.legend(fontsize=8)
        fig.suptitle('Fixed subsets · exact PREM moveout · 10–100 s · no phase-specific alignment\n'
                     'Group membership chosen from depth or coverage, not amplitude; smaller N itself raises finite-sample coherence',fontsize=11)
        save(fig,out/name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--broad-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();broad=args.broad_dir.resolve();out=args.output.resolve()
    if out.exists():parser.error('Output already exists')
    meta,sel,depths,d,inputs,time,predictions=load_context(broad);out.mkdir(parents=True)
    dump_json(out/'selection.json',sel)
    dump_json(out/'baseline_frozen_processing.json',json.loads((broad/'frozen_processing.json').read_text()))
    zoom(broad,out,meta,predictions)
    response,leakage=response_checks(broad,out,meta,sel,d,inputs,time,predictions)
    metrics,errors=coherence_checks(out,meta,sel,depths,d,inputs,time,predictions)
    dump_json(out/'provenance.json',dict(source_broad_directory=str(broad),source_provenance_sha256=hashlib.sha256((broad/'provenance.json').read_bytes()).hexdigest(),
        records=127,events=28,bands_s=BANDS,baseline_unchanged=True,
        zoom='Exact slices of saved envelopes; time -300:-100 s, p -1.5:+0.5 s/deg, common scale 0:0.08; hollow clouds.',
        controls='Three isolated positive unit impulses at individual SS/S400S/S670S predictions, filtered at native 0.125 s sampling then sampled to the saved broad input grid. Same 127 distances, shared physical-support mask, same broad_beam and padding. The common 10-100 s impulse peak is one; no per-band renormalization.',
        residuals='Primary diagnostic: signed correlation to same-model same-phase mean excluding the entire target event. Window -25:+25 s, lag +/-8 s, min signed correlation .7, peak margin .05, peak separation 5 s; template RMS and target standard deviation >=.001 baseline SS units. No polarity changes. Wide-lag absolute PREM-SS-wavelet fits are saved separately as exploratory; their PREM degradation exposes lobe/sign ambiguity.',
        timing_alignment_diagnostic='Only accepted fits receive a phase-specific shift; failed fits stay unshifted. All 127 records retained. Valid comparisons near central +/-25 s; diagnostic traces outside their shifted local support remain NaN. Fitting can select a different lobe or an interfering phase; stronger coherence after fitting is not independent phase validation.',
        groups='Fixed 0-70, 70-300, 300-700 km depth bins; six highest-count events with >=4 records. Same paired subsets in PREM/GLAD; no amplitude selection. Small-N coherence requires caution.',
        diagnostic_limitation='The signed +/-8 s fit tests modest sign-compatible timing scatter only. Larger delays, different lobes, phase interference and unknown 3D differential times remain possible.',
        interpretation='Controls isolate sampling/linear approximation. Exact-model correction tests those effects on data, not GLAD 3D ray corrections. Residual scatter, sign mismatch and waveform complexity can coexist; no new patch classified as an artifact or reflector.',
        processing_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(output=str(out),linear_approximation_errors=errors,
        coherence_10_100=[r for r in metrics if r['band_s']=='10-100'])),flush=True)

if __name__=='__main__':main()
