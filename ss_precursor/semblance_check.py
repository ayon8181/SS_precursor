"""Finite-window semblance diagnostics using summed beam and trace energies."""
import argparse
import json
from pathlib import Path
import numpy as np
from .circular_stacks import cap_weights
from .distance_section import load_global
from .pipeline import dump_json
from .stacking import sample_trace


def sample_terms(samples, weights):
    x = np.asarray(samples, dtype=float); w = np.asarray(weights, dtype=float)
    if (x.ndim != 2 or w.shape != (len(x),) or not np.isfinite(w).all()
            or np.any(w < 0) or not np.any(w > 0)):
        raise ValueError('Invalid samples or weights')
    valid = np.isfinite(x) & (w[:,None] > 0)
    effective = valid*w[:,None]
    values = np.where(valid, x, 0)
    return dict(A=np.sum(effective*values,axis=0), E=np.sum(effective*values**2,axis=0),
                W=np.sum(effective,axis=0), count=valid.sum(axis=0))


def centered_sum(values, length):
    x=np.asarray(values,dtype=float)
    if length < 1 or length % 2 != 1 or length > x.shape[-1]:
        raise ValueError('Window must have an odd number of samples within data length')
    c=np.concatenate((np.zeros(x.shape[:-1]+(1,)),np.cumsum(x,axis=-1)),axis=-1)
    sums=c[...,length:]-c[...,:-length]
    out=np.full(x.shape,np.nan)
    half=length//2
    out[...,half:x.shape[-1]-half] = sums
    return out


def window_semblance(A,E,W,count,length,minimum_support,energy_floor):
    """S=sum_t A(t)^2 / sum_t W(t)*E(t), never mean of instantaneous S.

    A=sum_i w_i x_i, E=sum_i w_i x_i^2, W=sum_i w_i, with identical
    finite-sample masks. Entire windows must satisfy the support threshold.
    Energy masking uses mean weighted trace energy; no epsilon is added.
    """
    A,E,W,count=map(np.asarray,(A,E,W,count))
    if not (A.shape==E.shape==W.shape==count.shape) or energy_floor < 0:
        raise ValueError('Invalid term shapes or energy threshold')
    valid=(count>=minimum_support)&(W>0)&np.isfinite(A)&np.isfinite(E)&np.isfinite(W)
    num=centered_sum(np.where(valid,A*A,0),length)
    den=centered_sum(np.where(valid,W*E,0),length)
    full=centered_sum(valid.astype(int),length)==length
    sum_w=centered_sum(np.where(valid,W,0),length)
    mean_energy=np.divide(centered_sum(np.where(valid,E,0),length),sum_w,
                          out=np.full(A.shape,np.nan),where=sum_w>0)
    good=full & (den>0) & (mean_energy>energy_floor)
    ratio=np.divide(num,den,out=np.full(A.shape,np.nan),where=good)
    return dict(semblance=ratio, numerator=num, denominator=den,
                mean_trace_energy=mean_energy, valid=good)


def dominant_period(tau,traces,weights,band):
    """Peak of averaged SS-window power, estimated independently of precursors."""
    keep=(tau>=-60)&(tau<=60); x=np.asarray(traces)[:,keep]
    dt=float(tau[1]-tau[0]); nfft=max(8192,2**int(np.ceil(np.log2(x.shape[1]))))
    x=(x-x.mean(axis=1,keepdims=True))*np.hanning(x.shape[1])
    power=np.average(abs(np.fft.rfft(x,n=nfft,axis=1))**2,weights=weights,axis=0)
    freq=np.fft.rfftfreq(nfft,dt); allowed=(freq>=1/max(band))&(freq<=1/min(band))
    peak=np.flatnonzero(allowed)[np.argmax(power[allowed])]
    return float(1/freq[peak]),freq,power


def plot_reference_radius(output,center,records):
    """Compact two-phase comparison at the radius nearest 900 km."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    radius=min((p['radius_km'] for p in records),key=lambda r:abs(r-900))
    chosen=[p for p in records if p['center_id']==center and p['radius_km']==radius]
    fig,axes=plt.subplots(len(chosen),4,figsize=(14,3.5*len(chosen)),constrained_layout=True,squeeze=False)
    for row,panel in enumerate(chosen):
        data=np.load(output/panel['output_file'])
        for col,(fraction,length,duration) in enumerate(zip(data['window_fractions'],data['window_samples'],data['window_durations_s'])):
            ax=axes[row,col]
            mesh=ax.pcolormesh(data['time_s'],data['residual_slowness'],data[f'window_{length}_semblance'],
                               shading='auto',vmin=0,vmax=1,cmap='viridis',rasterized=True)
            label='Instantaneous' if fraction==0 else f'{fraction:g} T ≈ {duration:.1f} s'
            ax.plot(0,0,'*',mfc='white',mec='black',ms=8)
            ax.set(xlabel='Time relative to phase (s)',ylabel='Residual slowness (s/degree)',
                   title=panel['phase'].replace('^','')+' · '+label,xlim=(-40,40),ylim=(-2,2))
    fig.colorbar(mesh,ax=axes.ravel().tolist(),shrink=.7,label='Semblance, common 0–1 scale')
    fig.suptitle(f'{center} · {radius:g} km · reference SS period {float(data["dominant_period_s"]):.1f} s\n'
                 'Numerator/denominator summed before division; full support and energy masks',fontsize=12)
    for ext in ['png','pdf']: fig.savefig(output/(center+'_900km_compact.'+ext),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vespagrams',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--energy-floor-fraction',type=float,default=1e-6)
    args=parser.parse_args()
    if not np.isfinite(args.energy_floor_fraction) or args.energy_floor_fraction<=0:
        parser.error('Energy floor fraction must be finite and positive')
    old=json.loads((args.vespagrams/'provenance.json').read_text())
    capdir=Path(old['source_caps']); cap=json.loads((capdir/'provenance.json').read_text())
    plan,_,members,tau,traces,rows,depths,_=load_global(old['bin_plan'],cap['distance_range_degree'])
    if plan['fingerprint'] != old['bin_plan_fingerprint'] or members != json.loads((capdir/'waveform_members.json').read_text()):
        raise ValueError('Source selection identity changed')
    reference=next(p for p in old['panels'] if p['radius_km']==min(old['radii_km'],key=lambda r:abs(r-900)))
    idx=reference['membership_indices']; weights=cap_weights(rows,idx,cap['weighting'])
    T,freq,power=dominant_period(tau,[traces[i] for i in idx],weights,cap['period_band_s'])
    dt=.5; fractions=[0,.25,.5,1.]
    lengths=[1 if f==0 else max(1,2*int(round(f*T/dt/2))+1) for f in fractions]
    padding=max(lengths)//2
    offsets=np.arange(-40-padding*dt,40+padding*dt+dt/2,dt)
    core=np.arange(padding,len(offsets)-padding)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spectrum,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    selected=(freq>=1/max(cap['period_band_s']))&(freq<=1/min(cap['period_band_s']))
    ax.plot(1/freq[selected],power[selected]/power[selected].max())
    ax.axvline(T,ls='--',color='black')
    ax.set(xlabel='Period (s)',ylabel='Relative averaged SS power',title=f'Reference SS spectral peak: {T:.1f} s')
    spectrum.savefig(output/'reference_SS_spectrum.png',dpi=180); plt.close(spectrum)
    records=[]
    for center in old['centers']:
        panels=[p for p in old['panels'] if p['center_id']==center]
        fig,axes=plt.subplots(len(panels),len(lengths),figsize=(14,2.5*len(panels)),constrained_layout=True,squeeze=False)
        for rownum,panel in enumerate(panels):
            idx=panel['membership_indices']; weights=cap_weights(rows,idx,cap['weighting'])
            phase=panel['phase']; distance=np.array([rows[i]['distance_degree'] for i in idx])
            prediction=np.array([rows[i]['arrivals'][phase]['time_s']-rows[i]['predicted_ss_s'] for i in idx])
            saved=np.load(args.vespagrams/panel['result_file']); q=saved['residual_slownesses']
            terms={key:[] for key in ['A','E','W','count']}
            for slow in q:
                samples=[]
                for i,origin,d in zip(idx,prediction,distance):
                    query=origin+offsets+slow*(d-old['reference_distance_degree'])
                    values=sample_trace(tau,traces[i],query)
                    values[abs(query-origin)>old['input_phase_half_window_s']]=np.nan
                    samples.append(values)
                point=sample_terms(samples,weights)
                for key in terms: terms[key].append(point[key])
            terms={key:np.asarray(value) for key,value in terms.items()}
            reference_energy=float(np.max(np.divide(terms['E'],terms['W'],out=np.zeros_like(terms['E']),where=terms['W']>0)))
            floor=max(1e-20,args.energy_floor_fraction*reference_energy)
            minimum=panel['minimum_contributing_traces']
            results={}
            for col,(fraction,length) in enumerate(zip(fractions,lengths)):
                computed=window_semblance(terms['A'],terms['E'],terms['W'],terms['count'],length,minimum,floor)
                sm=computed['semblance'][:,core]
                ax=axes[rownum,col]
                mesh=ax.pcolormesh(offsets[core],q,sm,shading='auto',vmin=0,vmax=1,cmap='viridis',rasterized=True)
                ax.plot(0,0,'*',mfc='white',mec='black',ms=7)
                label='Instantaneous' if fraction==0 else f'{fraction:g} T ≈ {length*dt:.1f} s'
                ax.set(title=f'{phase.replace("^", "")} · {panel["radius_km"]:g} km · {label}',
                       xlabel='Time relative to phase (s)',ylabel='Residual slowness (s/degree)',xlim=(-40,40),ylim=(-2,2))
                for key,value in computed.items(): results[f'window_{length}_{key}']=value[:,core]
                if fraction==0:
                    ok=np.isfinite(sm)
                    np.testing.assert_allclose(sm[ok],saved['semblance'][ok],atol=1e-10,rtol=1e-8)
            name=panel['result_file'].replace('.npz','_window_check.npz')
            np.savez_compressed(output/name,time_s=offsets[core],residual_slowness=q,
                dominant_period_s=T,window_samples=np.array(lengths),window_fractions=np.array(fractions),
                window_durations_s=np.array(lengths)*dt,energy_floor=floor,**results,
                **{key:value[:,core] for key,value in terms.items()})
            iz=int(np.argmin(abs(q))); it=int(np.argmin(abs(offsets[core])))
            records.append(dict(center_id=center,phase=phase,radius_km=panel['radius_km'],
                reference_energy=reference_energy,energy_floor=floor,minimum_support=minimum,
                at_prediction=[float(results[f'window_{n}_semblance'][iz,it]) for n in lengths],
                output_file=name))
        fig.colorbar(mesh,ax=axes.ravel().tolist(),shrink=.5,label='Semblance, common 0–1 scale')
        fig.suptitle(f'{center} · T={T:.1f} s from reference SS spectrum\n'
            'Sum numerator and denominator over time BEFORE division; full-window support and energy masks',fontsize=12)
        for ext in ['png','pdf']: fig.savefig(output/(center+'_window_comparison.'+ext),dpi=150)
        plt.close(fig)
    dump_json(output/'provenance.json',dict(source_vespagrams=str(args.vespagrams.resolve()),
        dominant_period_s=T,period_reference_panel=reference,window_fractions=fractions,
        window_samples=lengths,window_durations_s=[n*dt for n in lengths],
        energy_floor_fraction=args.energy_floor_fraction,absolute_energy_floor=1e-20,
        minimum_support_note='Every sample in the entire centred window must meet original panel support threshold.',
        reference_energy_note='Maximum weighted mean trace-square amplitude on padded panel scan; shared across window lengths.',
        formula='sum_t (sum_i w_i*x_i)^2 / sum_t [(sum_i w_i)*(sum_i w_i*x_i^2)]',
        no_epsilon=True,window_edges='Full windows required; padded waveform queries computed, not reflected images.',
        signed_stack_note='Original signed stacks and envelopes are unchanged.',panels=records))
    for center in old['centers']: plot_reference_radius(output,center,records)
    print(json.dumps(dict(output=str(output),dominant_period_s=T,window_durations_s=[n*dt for n in lengths],panels=len(records))))


if __name__=='__main__': main()
