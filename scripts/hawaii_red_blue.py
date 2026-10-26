from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'outputs/Hawaii_900km_comparison'
OUT=ROOT;OUT.mkdir(parents=True,exist_ok=True)
records=json.loads((ROOT/'accepted_records.json').read_text());points=json.loads((ROOT/'predictions.json').read_text())
wave=np.load(ROOT/'padded_filtered_waveforms.npy',mmap_mode='r');time=np.load(ROOT/'input_time_s.npy')
d=np.array([r['distance_degree'] for r in records]);edges=np.arange(100.,161.);centers=(edges[:-1]+edges[1:])/2
populations={'SY only':np.flatnonzero([r['station'].startswith('SY.') for r in records]),'All stations':np.arange(len(records))}
use=(time>=-350)&(time<=50);t=time[use];sections={};counts={};coherence=[]
for label,idx in populations.items():
    counts[label]=np.histogram(d[idx],edges)[0]
    sec=np.full((2,60,len(t)),np.nan)
    for j in range(60):
        ii=idx[(d[idx]>=edges[j])&(d[idx]<edges[j+1])]
        if len(ii):sec[:,j,:]=np.nanmean(wave[:,ii,:][:,:,use],axis=1)
    sections[label]=sec
    for mi,model in enumerate(['PREM','GLAD-M35']):
        for phase in ['S^400S','S^670S']:
            offset=np.arange(-25.,26.)
            rows=[]
            for i in idx:
                rel=records[i]['arrivals'][phase]['time_s']-records[i]['arrivals']['SS']['time_s']
                rows.append(np.interp(rel+offset,time,wave[mi,i],left=np.nan,right=np.nan))
            u=np.array(rows);valid=np.isfinite(u);n=valid.sum(axis=0);v=np.where(valid,u,0)
            den=float(np.sum(n*np.sum(v*v,axis=0)));num=float(np.sum(np.sum(v,axis=0)**2))
            coherence.append(dict(population=label,model=model,phase=phase,records=len(idx),window_s=[-25,25],integrated_semblance=num/den if den else None))
for bounds,stem in [((-350,50),'distance_stacks_red_blue'),((-280,-110),'distance_stacks_red_blue_precursors')]:
    fig=plt.figure(figsize=(13,11),constrained_layout=True)
    gs=fig.add_gridspec(3,2,height_ratios=[.65,4,4]);axes=[]
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('.65')
    for ci,(label,idx) in enumerate(populations.items()):
        countax=fig.add_subplot(gs[0,ci]);countax.stairs(counts[label],edges,color='black',lw=1)
        countax.set(xlim=(100,160),ylabel='Records / 1°',title='%s | %d records, %d events'%(label,len(idx),len({records[i]['event_id'] for i in idx})))
        countax.tick_params(labelbottom=False)
        for mi,model in enumerate(['PREM','GLAD-M35']):
            ax=fig.add_subplot(gs[mi+1,ci]);axes.append(ax)
            mask=(t>=bounds[0])&(t<=bounds[1])
            im=ax.pcolormesh(centers,t[mask],np.ma.masked_invalid(sections[label][mi][:,mask].T),shading='nearest',cmap=cmap,vmin=-.08,vmax=.08,rasterized=True)
            selected=set(idx.tolist())
            for phase in ['S^400S','S^670S']:
                pts=[r for r in points if r['phase']==phase and r['index'] in selected]
                for j in range(60):
                    local=[r for r in pts if edges[j]<=d[r['index']]<edges[j+1]]
                    if not local:continue
                    predicted=np.array([r['time_at_record_distance_s'] for r in local])
                    ax.vlines(centers[j],predicted.min(),predicted.max(),color='.3',lw=.4,alpha=.5)
                    ax.scatter([centers[j]],[np.median(predicted)],s=5,facecolors='none',edgecolors='.3',linewidths=.4)
            if bounds[1]>0:ax.axhline(0,color='black',ls=':',lw=.65)
            ax.set(xlim=(100,160),ylim=(bounds[1],bounds[0]),title=model)
            if mi==1:ax.set_xlabel('Epicentral distance (degree)')
            if ci==0:ax.set_ylabel('Time relative to PREM-predicted SS (s)')
    fig.colorbar(im,ax=axes,label='Signed mean / baseline SS amplitude',shrink=.8,extend='both')
    fig.suptitle('Hawaii (19.5°N, 155.5°W), 900 km | Fixed 1° distance means\n10–100 s, 2 corners; theoretical SS alignment; PREM polarity for both models\nSame amplitude scale in every panel; gray = empty bin; hollow points = median PREM predictions per bin; whiskers = prediction ranges',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(OUT/(stem+'.'+ext),dpi=200)
    plt.close(fig)
np.savez_compressed(OUT/'distance_stacks_red_blue.npz',time_relative_SS_s=t,distance_edges_degree=edges,populations=np.array(list(populations)),models=np.array(['PREM','GLAD-M35']),signed_mean=np.array(list(sections.values())),record_count=np.array(list(counts.values())))
(OUT/'precursor_coherence.json').write_text(json.dumps(coherence,indent=2))
(OUT/'red_blue_notes.txt').write_text('Fixed nonmoving 1-degree signed distance means, equal trace weights; each bin divided by its valid record count. Original Hawaii selected records and normalization unchanged. No extra filtering, polarity changes, or timing shifts. Common color scale +/-0.08 baseline SS amplitude; SS saturates, empty bins gray.\nCoherence diagnostic: sum_t(sum_i u_i)^2 / sum_t(N_t sum_i u_i^2), in +/-25 s around each record PREM-predicted phase time. Exact PREM phase moveout is used only for this scalar diagnostic, not the distance images. These values are not significance estimates; denser arrays overweight a few events, and finite record counts affect comparisons.\n')
print(json.dumps(coherence,indent=2));print(OUT)
