"""Reproducible CPU benchmark on non-document synthetic proxies.

Real YOLO is included by default; --supplied measures analysis with supplied
localization (including a forced MRZ-detector miss). No personal images saved.
"""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from passport_quality_gate.runtime_metrics import memory_mib
import torch

from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.viewport import CaptureViewport
from copy import deepcopy
from passport_quality_gate.synthetic import sample, GUIDE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=40)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--supplied',action='store_true')
    parser.add_argument('--viewport-v4',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(args.threads); cv2.setNumThreads(1)
    image,det,_=sample()
    class Supplied:
        def locate(self,frame,mode='preview'):
            return det
    report={'machine':platform.platform(),'cpu':platform.processor(),'logical_cpus':__import__('os').cpu_count(),
            'torch_threads':args.threads,'opencv_threads':1,'python':platform.python_version(),
            'torch':torch.__version__,'opencv':cv2.__version__,'device':'cpu','vram':None,
            'workload':'synthetic non-document 1024x768; supplied localization' if args.supplied else
                       'synthetic non-document 1024x768; real packaged YOLO every frame',
            'warmup_frames':args.warmup,'measured_frames':args.frames,'profiles':{}}
    viewport=CaptureViewport(.16,.18,.68,.64)
    roi,viewport_meta=viewport.extract(image)
    report['viewport']=viewport_meta if args.viewport_v4 else None
    report['selector_frame_bytes']={'raw':image.nbytes,'roi':roi.nbytes} if args.viewport_v4 else None
    for profile in (('fp2','vnext','v4') if args.viewport_v4 else ('fp2','vnext')):
        start=perf_counter()
        selected_det=deepcopy(det)
        if profile=='v4':
            offset=np.array(viewport_meta['viewport_pixel_rect'][:2])
            selected_det.polygon-=offset; selected_det.mrz_polygon-=offset
        class CurrentSupplied:
            def locate(self,frame,mode='preview'): return selected_det
        config={'vnext':'configs/research_vnext.yaml','v4':'configs/research_v4.yaml'}.get(profile)
        gate=PassportQualityGate(config=config,device='cpu',localizer=CurrentSupplied() if args.supplied else None)
        def analyze(t):
            if profile=='v4':
                t0=perf_counter(); selected_roi,meta=viewport.extract(image)
                crop_ms=(perf_counter()-t0)*1000
                result=gate.analyze_roi_preview(selected_roi,timestamp=t,viewport_metadata=meta)
                result['timing_ms']['viewport_extract']=crop_ms
                result['timing_ms']['total']+=crop_ms
                return result
            return gate.analyze_preview(image,GUIDE,timestamp=t)
        startup=(perf_counter()-start)*1000
        for i in range(args.warmup):
            analyze((i+1)*.1)
        times=[]; ram=[]; found=0
        start=perf_counter()
        for i in range(args.frames):
            result=analyze((args.warmup+i+1)*.1)
            times.append(result['timing_ms']); ram.append((memory_mib() or 0.))
            found+=int(result['passport_found'])
        elapsed=perf_counter()-start
        stages={key:np.percentile([t[key] for t in times if key in t],[50,95,99]).tolist()
                for key in sorted(set().union(*(t.keys() for t in times)))}
        report['profiles'][profile]={'startup_ms':startup,'latency_ms_p50_p95_p99':stages.pop('total'),
                                    'stage_ms_p50_p95_p99':stages,'analysis_hz':args.frames/elapsed,
                                    'process_peak_sampled_rss_mib':max(ram),'page_found_frames':found}
    if args.supplied:
        det.mrz_polygon=None; det.mrz_confidence=0.
        gate=PassportQualityGate(config='configs/research_vnext.yaml',localizer=Supplied(),device='cpu')
        times=[]
        for i in range(args.frames+args.warmup):
            r=gate.analyze_preview(image,GUIDE,timestamp=(i+1)*.1)
            if i>=args.warmup: times.append(r['timing_ms']['mrz_presence'])
        report['forced_mrz_miss_fallback_ms_p50_p95_p99']=np.percentile(times,[50,95,99]).tolist()
        if args.viewport_v4:
            from passport_quality_gate.config import load_config
            misses={}
            for enabled in (False,True):
                cfg=load_config('configs/research_v4.yaml');cfg['research']['fallback_telemetry']=enabled
                selected_det=deepcopy(det);selected_det.polygon-=np.array(viewport_meta['viewport_pixel_rect'][:2])
                gate=PassportQualityGate(config=cfg,localizer=CurrentSupplied(),device='cpu');times=[]
                for i in range(args.frames+args.warmup):
                    r=gate.analyze_roi_preview(roi,timestamp=(i+1)*.1)
                    if i>=args.warmup: times.append(r['timing_ms']['mrz_presence'])
                misses[str(enabled)]=np.percentile(times,[50,95,99]).tolist()
            report['v4_forced_miss_telemetry_ms_p50_p95_p99']=misses
    payload=json.dumps(report,indent=2,allow_nan=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload+'\n',encoding='utf-8')


if __name__=='__main__': main()
