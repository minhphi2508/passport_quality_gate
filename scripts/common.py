from pathlib import Path
import sys, argparse, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2, yaml
from passport_quality_gate import Analyzer, YoloLocalizer, ProxyLocalizer
from passport_quality_gate.io import load_image, load_detection, save_result

def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',type=Path,default=ROOT/'configs/thresholds.yaml')
    p.add_argument('--backend',choices=['yolo','proxy','oracle'],default='yolo',help='proxy/oracle are synthetic experiments only')
    p.add_argument('--weights',type=Path,default=ROOT/'models/passport_detector_ver3_best.pt')
    p.add_argument('--device',default='cpu')
    p.add_argument('--mode',choices=['preview','final'],default='final')
    p.add_argument('--guide',default='0.16,0.18,0.68,0.64',help='normalized x,y,w,h; caller must align to frame')
    p.add_argument('--threads',type=int,default=2)
    return p

def build(args):
    cv2.setNumThreads(args.threads)
    cfg=yaml.safe_load(args.config.read_text())
    if args.backend=='yolo':
        import torch
        torch.set_num_threads(args.threads)
        localizer=YoloLocalizer(args.weights,cfg['localization'],args.device)
    else: localizer=ProxyLocalizer()
    guide=dict(zip(('x','y','w','h'),map(float,args.guide.split(','))))
    return Analyzer(localizer,cfg),guide

def run_one(analyzer,guide,path,args,timestamp=None):
    frame=load_image(path); det=None
    if args.backend=='oracle':
        meta=json.loads(Path(path).with_suffix('.json').read_text())
        if meta['detection']['source']!='synthetic_ground_truth': raise ValueError('Oracle mode only accepts synthetic metadata')
        det=load_detection(meta); guide=meta['guide_box']
    return frame,analyzer.analyze_frame(frame,guide,args.mode,timestamp=timestamp,detection=det)
