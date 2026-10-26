"""Matched PREM/GLAD comparison for Hawaii; all stations versus SY only."""
from pathlib import Path
from collections import Counter,defaultdict
import concurrent.futures,json,sys,xml.etree.ElementTree as ET
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ss_precursor.geometry import spherical_geometry
from ss_precursor.circular_stacks import xyz
from ss_precursor.arrivals import unique_reflection
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/Hawaii_900km_comparison'
CENTER=(19.5,-155.5);RADIUS=900.

def tau_init():
    global MODEL
    from obspy.taup import TauPyModel
    MODEL=TauPyModel('prem')

def phase_selection(row):
    try:
        rays={}
        for phase,bound in [('S^400S',400),('S^670S',670)]:
            a=unique_reflection(MODEL,row['depth_km'],row['distance_degree'],phase,bound,row['event_latitude'],row['event_longitude'],row['azimuth_degree'])
            arc=spherical_geometry(*CENTER,a['reflection_latitude'],a['reflection_longitude'])[0]
            if arc*6371*np.pi/180>RADIUS:return None
            rays[phase]=a
        rays['SS']=unique_reflection(MODEL,row['depth_km'],row['distance_degree'],'SS',0,row['event_latitude'],row['event_longitude'],row['azimuth_degree'])
        row['arrivals']=rays
        return row
    except Exception as exc:
        return dict(error=str(exc),event_id=row['event_id'],station=row['station'])

def main():
    OUT.mkdir(exist_ok=True)
    plans={m:json.loads((ROOT/'outputs'/d/'events.json').read_text()) for m,d in [('PREM','prem_event_reference'),('GLAD','glad_m35_event_reference')]}
    for m,d in [('PREM','prem_event_reference'),('GLAD','glad_m35_event_reference')]:
        plans[m]['cache_root']=json.loads((ROOT/'outputs'/d/'bin_plan/plan.json').read_text())['cache_root']
    gm={r['event_id']:r for r in plans['GLAD']['tasks']}
    if (OUT/'geometric_selection.json').exists():
        selected=json.loads((OUT/'geometric_selection.json').read_text())
    else:
        candidates=[];counts=Counter();cv=xyz([CENTER])[0]
        for ni,task in enumerate(plans['PREM']['tasks']):
            event_id=task['event_id']
            if event_id not in gm:continue
            gt=gm[event_id]
            prov=json.loads((Path(plans['PREM']['cache_root'])/'events'/('event_%06d'%task['index'])/'provenance.json').read_text())
            evlat,evlon,depth=prov['origin_latitude'],prov['origin_longitude'],prov['origin_depth_km']
            ev=xyz([[evlat,evlon]])[0]
            with h5py.File(task['source'],'r') as pf,h5py.File(gt['source'],'r') as gf:
                common=sorted(set(pf['Waveforms'])&set(gf['Waveforms']))
                for station in common:
                    counts['common_event_station_entries']+=1
                    try:
                        xml=ET.fromstring(pf['Waveforms'][station]['StationXML'][()].tobytes())
                        node=xml.find('.//{*}Station');lat=float(node.find('{*}Latitude').text);lon=float(node.find('{*}Longitude').text)
                        distance,az,baz=spherical_geometry(evlat,evlon,lat,lon)
                        if not 100<=distance<=160:continue
                        # Necessary geometric condition only: any ray bounce must
                        # lie on this great circle. This screen excludes no cap rays.
                        sv=xyz([[lat,lon]])[0];normal=np.cross(ev,sv);normal/=np.linalg.norm(normal)
                        cross_track=np.arcsin(np.clip(abs(np.dot(normal,cv)),0,1))*6371
                        if cross_track>RADIUS+1e-6:continue
                        candidates.append(dict(event_id=event_id,station=station,station_latitude=lat,station_longitude=lon,event_latitude=evlat,event_longitude=evlon,depth_km=depth,distance_degree=distance,azimuth_degree=az,backazimuth_degree=baz,PREM_source=task['source'],GLAD_source=gt['source'],PREM_task=task['index'],GLAD_task=gt['index']))
                    except Exception as exc:
                        counts['inventory_errors']+=1
            print('Inventory %d/%d; ray candidates %d'%(ni+1,len(plans['PREM']['tasks']),len(candidates)),flush=True)
        (OUT/'inventory_summary.json').write_text(json.dumps(dict(counts),indent=2))
        selected=[];errors=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=6,initializer=tau_init) as pool:
            for i,result in enumerate(pool.map(phase_selection,candidates,chunksize=12)):
                if result is not None:
                    if 'error' in result:errors.append(result)
                    else:selected.append(result)
                if (i+1)%200==0:print('Rays %d/%d; selected %d'%(i+1,len(candidates),len(selected)),flush=True)
        (OUT/'geometric_selection.json').write_text(json.dumps(selected,indent=2))
        (OUT/'ray_rejections.json').write_text(json.dumps(errors,indent=2))
    print('GEOMETRY',len(selected),'SY',sum(r['station'].startswith('SY.') for r in selected),'events',len({r['event_id'] for r in selected}),'networks',dict(Counter(r['station'].split('.')[0] for r in selected)),flush=True)
    process(selected,plans)


def process(selected,plans):
    from ss_precursor.asdf_io import open_asdf,select_event,associated_stream
    from ss_precursor.preprocessing import transverse,align
    from ss_precursor.broad_vespagrams import filter_native,broad_beam
    if (OUT/'accepted_records.json').exists():
        records=json.loads((OUT/'accepted_records.json').read_text());wave=np.load(OUT/'padded_filtered_waveforms.npy');query=np.load(OUT/'input_time_s.npy')
    else:
        # Wide input support for exactly the same padded envelope calculation.
        query=np.arange(-1786.,1587.);byevent=defaultdict(list)
        for row in selected:byevent[row['event_id']].append(row)
        records=[];paired=[];rejected=[]
        for gi,(event_id,rows) in enumerate(byevent.items()):
            with open_asdf(Path(rows[0]['PREM_source'])) as pd,open_asdf(Path(rows[0]['GLAD_source'])) as gd:
                datasets={'PREM':pd,'GLAD':gd};origins={}
                for m in datasets:
                    _,origins[m]=select_event(datasets[m],event_id,plans[m]['configuration']['origin'])
                for original in rows:
                    row=dict(original)
                    try:
                        traces={};amplitudes={};prem_sign=None;trace_meta={}
                        for m in ['PREM','GLAD']:
                            cfg=plans[m]['configuration'];ds=datasets[m];group=ds.waveforms[row['station']];coords=group.coordinates
                            if not np.allclose([coords['latitude'],coords['longitude']],[row['station_latitude'],row['station_longitude']],atol=1e-6,rtol=0):raise ValueError('Model station coordinates differ')
                            o=origins[m]
                            if not np.allclose([o.latitude,o.longitude,o.depth/1000],[row['event_latitude'],row['event_longitude'],row['depth_km']],atol=1e-6,rtol=0):raise ValueError('Model source geometry differs')
                            stream=associated_stream(group,cfg['waveform_tag'],event_id,len(ds.events));tr=transverse(stream,row['backazimuth_degree'],cfg['horizontal_convention'])
                            # Existing baseline normalization and PREM sign convention,
                            # but never use the measured maximum to shift time.
                            _,_,measured=align(tr,o,row['arrivals']['SS']['time_s'],cfg)
                            if m=='PREM':prem_sign=measured['polarity_multiplier']
                            traces[m]=tr;amplitudes[m]=measured['ss_amplitude_mm']
                            trace_meta[m]=dict(trace_id=tr.id,SS_amplitude_mm=measured['ss_amplitude_mm'],trace_start_from_origin_s=float(tr.stats.starttime-o.time),measured_SS_for_amplitude_only_s=measured['measured_ss_s'])
                        data=[]
                        for m in ['PREM','GLAD']:
                            values=filter_native(traces[m],origins[m],plans[m]['configuration'],(10,100),row['arrivals']['SS']['time_s'],amplitudes[m],prem_sign,query)
                            if not np.isfinite(values[(query>=-500)&(query<=300)]).all():raise ValueError('Insufficient central waveform support')
                            data.append(values)
                        if trace_meta['PREM']['trace_id']!=trace_meta['GLAD']['trace_id']:raise ValueError('Model trace IDs differ')
                        row.update(trace_metadata=trace_meta,PREM_polarity_multiplier=prem_sign)
                        records.append(row);paired.append(data)
                    except Exception as exc:
                        rejected.append(dict(event_id=event_id,station=row['station'],reason=str(exc)))
            print('Processed event %d/%d; pairs %d'%(gi+1,len(byevent),len(records)),flush=True)
        wave=np.array(paired).transpose(1,0,2)
        (OUT/'accepted_records.json').write_text(json.dumps(records,indent=2));(OUT/'processing_rejections.json').write_text(json.dumps(rejected,indent=2))
        np.save(OUT/'padded_filtered_waveforms.npy',wave);np.save(OUT/'input_time_s.npy',query)
    if not records:raise ValueError('No matched Hawaii records')
    d=np.array([r['distance_degree'] for r in records]);reference=float(d.mean())
    populations={'SY_only':np.array([i for i,r in enumerate(records) if r['station'].startswith('SY.')]),'all_stations':np.arange(len(records))}
    predictions=[]
    for i,r in enumerate(records):
        ss=r['arrivals']['SS']
        for phase in ['S^400S','S^670S']:
            a=r['arrivals'][phase];p=a['ray_parameter_s_degree']-ss['ray_parameter_s_degree'];rel=a['time_s']-ss['time_s']
            predictions.append(dict(index=i,phase=phase,time_at_record_distance_s=rel,projected_time_at_reference_s=rel-p*(r['distance_degree']-reference),relative_slowness_s_degree=p))
    (OUT/'predictions.json').write_text(json.dumps(predictions,indent=2))
    summary={}
    for pop,idx in populations.items():
        folder=OUT/pop;folder.mkdir(exist_ok=True)
        if not len(idx):continue
        dd=d[idx];robust=float(np.diff(np.percentile(dd,[10,90]))[0])
        summary[pop]=dict(records=len(idx),events=len({records[i]['event_id'] for i in idx}),networks=dict(Counter(records[i]['station'].split('.')[0] for i in idx)),reference_distance_degree=reference,distance_range_degree=[float(dd.min()),float(dd.max())],nominal_slowness_scale_s_degree=float(np.sqrt(1000)/robust) if robust>0 else None)
        summary[pop]['event_record_counts']=dict(Counter(records[i]['event_id'] for i in idx))
        summary[pop]['unique_station_count']=len({records[i]['station'] for i in idx})
        for mi,model in enumerate(['PREM','GLAD']):
            path=folder/(model+'_10_100s.npz')
            if not path.exists():
                result=broad_beam(query,wave[mi,idx],dd,np.arange(-350.,51.),np.arange(-60,61)*.05,reference,(10,100),padding=1200.)
                np.savez_compressed(path,time_s=np.arange(-350.,51.),relative_slowness_s_degree=np.arange(-60,61)*.05,reference_distance_degree=reference,**result)
            print('Beam',pop,model,flush=True)
    (OUT/'coverage.json').write_text(json.dumps(summary,indent=2));plot(records,wave,query,predictions,populations,summary)
    (OUT/'README.txt').write_text('Hawaii comparison centered at 19.5 N, 155.5 W, radius 900 km. Both PREM S400S and S670S bounce points must be inside the cap; distances 100–160 degrees.\nSY-only versus all available networks; all includes SY and is not restricted to USArray. Same matched event/station records for PREM and GLAD in each population. Same 54-event PREM source list; no extra GLAD events.\nAll traces aligned to PREM theoretical SS arrivals, retaining 3D timing deviations. Identical 10–100 s, 2-corner forward-reverse filter; baseline detrend/taper/padding/edge guard. Individual model SS absolute amplitudes normalize traces; PREM SS-window sign multiplier applied to both. Measured peaks are used only for normalization/sign, never time alignment. Original +5 s timestamp included.\nTrace-weighted signed means divide by support, not sums. Shared reference distance is the all-station mean distance; both populations use the same time/slowness coordinates. Common envelope scale 0–0.08; masked gray, zero white. Slowness resolution is limited by the aperture; see coverage.json.\nInput geographic screen is a necessary great-circle cross-track condition, followed by exact PREM ray bounce selection; no midpoint approximation.\nRecord sections show individual traces without distance averages; same display gain in every panel.\nSix worker processes used for independent TauP geometry only.\n')
    print('FINAL',json.dumps(summary),flush=True)


def plot(records,wave,query,points,populations,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap=plt.get_cmap('Greys').copy();cmap.set_bad('.65')
    for zoom,stem in [(False,'vespagrams'),(True,'vespagrams_precursors')]:
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True,constrained_layout=True)
        for ci,(pop,idx) in enumerate(populations.items()):
            if not len(idx):continue
            pp=[r for r in points if r['index'] in set(idx.tolist())]
            for mi,model in enumerate(['PREM','GLAD']):
                ax=axes[mi,ci];a=np.load(OUT/pop/(model+'_10_100s.npz'))
                t,p=a['time_s'],a['relative_slowness_s_degree'];ti=(t>=(-280 if zoom else -350))&(t<=(-110 if zoom else 50));pi=(p>=(-1.2 if zoom else -3))&(p<=(.4 if zoom else 3))
                v=np.ma.array(a['envelope'],mask=~a['envelope_valid'])[np.ix_(pi,ti)]
                ax.set_facecolor('.65');im=ax.contourf(t[ti],p[pi],v,levels=np.linspace(0,.08,9),cmap=cmap,extend='max')
                for phase,label in [('S^400S','S400S'),('S^670S','S670S')]:
                    rows=[r for r in pp if r['phase']==phase];x=[r['projected_time_at_reference_s'] for r in rows];y=[r['relative_slowness_s_degree'] for r in rows]
                    ax.plot([min(x),max(x),max(x),min(x),min(x)],[min(y),min(y),max(y),max(y),min(y)],color='black',lw=.6)
                    ax.annotate(label,(np.median(x),max(y)),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
                counts=summary[pop];ax.set_title(('%s | %s\nN=%d, events=%d')%('PREM' if model=='PREM' else 'GLAD-M35',pop.replace('_',' '),len(idx),counts['events']),fontsize=10)
                if mi==1:ax.set_xlabel('Time relative to PREM-predicted SS at D₀ (s)')
                if ci==0:ax.set_ylabel('Slowness relative to SS (s/degree)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Envelope / baseline SS amplitude',shrink=.8)
        fig.suptitle('Hawaii (19.5°N, 155.5°W), 900 km | 10–100 s, 2 corners\nTheoretical SS alignment; PREM polarity for both models; common precursor scale',fontsize=11)
        for ext in ['png','pdf']:fig.savefig(OUT/(stem+'.'+ext),dpi=200)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,14),sharex=True,sharey=True,constrained_layout=True)
    use=(query>=-350)&(query<=50)
    for ci,(pop,idx) in enumerate(populations.items()):
        for mi,name in enumerate(['PREM','GLAD-M35']):
            ax=axes[mi,ci]
            for i in sorted(idx,key=lambda i:records[i]['distance_degree']):
                ax.plot(query[use],records[i]['distance_degree']-.2*wave[mi,i,use],color='black',lw=.35,alpha=.8)
            ax.axvline(0,color='.5',ls=':',lw=.6)
            ax.set(title='%s | %s | N=%d'%(name,pop.replace('_',' '),len(idx)),xlim=(-350,50),ylim=(161,99))
            if mi==1:ax.set_xlabel('Time relative to PREM-predicted SS (s)')
            if ci==0:ax.set_ylabel('Distance (degree)')
    fig.suptitle('Hawaii, 900 km | Individual waveforms, no distance averaging\n10–100 s; same gain 0.2° per baseline-SS amplitude; PREM polarity retained',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(OUT/('individual_record_sections.'+ext),dpi=200)
    plt.close(fig)

if __name__=='__main__':main()
