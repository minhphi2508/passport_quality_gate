"""V3 standalone runner; leaves existing run_webcam.py untouched.
CPU by default. C checks the current unannotated frame; only --save-captures
writes image pixels. Preview READY always requires a final check.
"""
from common import *
import numpy as np
import math
from time import monotonic, strftime
MESSAGES = {'BLUR': 'Anh mo: giu yen / cho lay net',
 'CROPPED': 'Dua day du trang vao anh',
 'GLARE': 'Tranh anh sang phan chieu',
 'HOLD_STEADY': 'Giu yen de xac nhan chat luong',
 'LOCALIZATION_UNCERTAIN': 'Chua xac minh duoc bien trang',
 'LOW_CONTRAST': 'Chu mo nhat: dieu chinh anh sang',
 'LOW_RESOLUTION': 'Chu qua nho: dua camera gan hon',
 'MOVE_DOWN': 'Dich passport xuong',
 'MOVE_LEFT': 'Dich passport sang trai',
 'MOVE_RIGHT': 'Dich passport sang phai',
 'MOVE_UP': 'Dich passport len',
 'MRZ_NOT_FOUND': 'Can thay ro ca hai dong MRZ',
 'NOISE': 'Anh nhieu: tang anh sang va giu yen',
 'PASSPORT_NOT_FOUND': 'Dua trang passport vao khung',
 'PASSPORT_TOO_CLOSE': 'Dua passport xa hon',
 'PASSPORT_TOO_FAR': 'Dua passport gan hon',
 'PERSPECTIVE_TOO_HIGH': 'Giu camera song song voi trang',
 'QUALITY_UNCERTAIN': 'Chua du chi tiet chu de danh gia',
 'READY': 'San sang chup (kiem tra so bo)',
 'ROTATED': 'Xoay passport dung chieu',
 'TOO_BRIGHT': 'Giam phoi sang',
 'TOO_DARK': 'Tang anh sang'}

MESSAGES = MESSAGES | {
 'CAMERA_RESOLUTION_LOW':'Thieu pixel: tang do phan giai / doi camera',
 'LOCALIZATION_UNCERTAIN':'Chua ro 4 canh: doi nen / tranh phan chieu',
 'READY':'Co the chup; can kiem tra anh cuoi',
}


def main():
 p=parser('V3 camera/video smoke test; CPU default')
 p.add_argument('--source',default='0',help='Camera index, video file or stream URL')
 p.add_argument('--capture-backend',choices=['auto','dshow','msmf'],default='auto')
 p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
 p.add_argument('--fps',type=float,default=4)
 p.add_argument('--headless',action='store_true');p.add_argument('--max-frames',type=int,default=0)
 p.add_argument('--log',type=Path)
 p.add_argument('--save-captures',type=Path,help='Opt in to saving each C capture and debug images locally')
 p.set_defaults(mode='preview',config=ROOT/'configs/thresholds_v3.yaml');a=p.parse_args()
 if a.backend=='oracle':p.error('Oracle belongs in synthetic tests only')
 if not math.isfinite(a.fps) or a.fps<=0 or min(a.width,a.height)<32:p.error('Invalid fps or dimensions')
 analyzer,guide=build(a);camera=a.source.isdigit()
 source=int(a.source) if camera else a.source
 backend={'auto':cv2.CAP_ANY,'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF}[a.capture_backend]
 cap=cv2.VideoCapture(source,backend)
 if not cap.isOpened():p.error('Cannot open source; try --capture-backend dshow or --source 1')
 if camera:
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height)
 log=None
 if a.log:
  a.log.parent.mkdir(parents=True,exist_ok=True);log=a.log.open('w',encoding='utf-8')
 last=-1e10;r=None;processed=0;index=0;final_text='C: kiem tra anh cuoi | R: reset | Q: thoat';capture_index=0
 source_fps=cap.get(cv2.CAP_PROP_FPS)
 if not math.isfinite(source_fps) or source_fps<=0:source_fps=30.
 def record(result):
  result['capture_metadata']={'requested_width':a.width,'requested_height':a.height,
      'backend':a.capture_backend,'source_kind':'camera' if camera else 'video_or_stream'}
  if log:log.write(json.dumps(result,allow_nan=False)+'\n');log.flush()
 try:
  while True:
   ok,raw=cap.read()
   if not ok:break
   clock=monotonic();t=clock if camera or '://' in a.source else index/source_fps;index+=1
   if index==1:print(f'Actual frame: {raw.shape[1]}x{raw.shape[0]}; requested {a.width}x{a.height}',flush=True)
   if t-last>=1/a.fps-1e-9:
    last=t;r=analyzer.analyze_frame(raw,guide,'preview',timestamp=t);processed+=1;record(r)
   if not a.headless:
    display=raw.copy();color=(0,190,255)
    if r:
     fresh=t-last<1.;ready=r['ready_for_capture'] and fresh
     color=(40,220,40) if ready else color
     cv2.polylines(display,[np.array(r['guide_polygon'],np.int32)],True,color,2)
     loc=r['localization']
     if fresh and loc['polygon'] is not None:
      cv2.polylines(display,[np.array(loc['polygon'],np.int32)],True,(255,180,50),1)
     code=r.get('guidance_code') or 'HOLD_STEADY'
     text=MESSAGES.get(code,code) if fresh else 'Dang doi ket qua moi'
     def score(name):
      v=r['quality'].get(name);return '?' if v is None else f'{v:.2f}'
     ce=r.get('quality_evidence',{})
     lines=[text,
       f"{r['state'] if fresh else 'WAIT'} | blur {score('blur_score')} | resolution {score('low_resolution_score')} | glare {score('glare_score')}",
       f"corners {ce.get('corners_verified',False)} | MRZ {ce.get('mrz_present',False)} | {raw.shape[1]}x{raw.shape[0]} | {r['timing_ms']['total']:.0f} ms",
       'Block: '+(', '.join(r['blocking_issues']) or 'none; checking stability'),final_text]
    else:lines=['Dang khoi dong',final_text]
    # Header is outside camera pixels; it cannot hide the top edge of the page.
    scale=min(1.,960/display.shape[1]);display=cv2.resize(display,None,fx=scale,fy=scale)
    panel=np.full((150,max(720,display.shape[1]),3),20,np.uint8)
    for j,line in enumerate(lines):cv2.putText(panel,line,(10,24+j*27),cv2.FONT_HERSHEY_SIMPLEX,.48,color if j==0 else (235,235,235),1,cv2.LINE_AA)
    canvas=np.zeros((display.shape[0],panel.shape[1],3),np.uint8);canvas[:,:display.shape[1]]=display
    cv2.imshow('Passport Quality Gate V3',np.vstack([panel,canvas]))
    key=cv2.waitKey(1)&255
    if key==ord('q'):break
    if key==ord('r'):analyzer.reset();r=None;last=-1e10
    if key==ord('c'):
     # This checks source pixels, never a drawn overlay or upscaled substitute.
     result=analyzer.analyze_frame(raw.copy(),guide,'final');record(result)
     final_text='FINAL: '+result['state']+' | '+(result['primary_issue'] or 'provisional quality pass')
     print(final_text,flush=True)
     if a.save_captures:
      capture_index+=1;out=a.save_captures/(strftime('%Y%m%d_%H%M%S')+f'_{capture_index:03}')
      out.mkdir(parents=True,exist_ok=True)
      cv2.imwrite(str(out/'capture.png'),raw)
      save_result(out/'final',raw,result,analyzer.debug)
     analyzer.reset();r=None;last=-1e10
   if a.max_frames and processed>=a.max_frames:break
 finally:
  cap.release()
  if log:log.close()
  if not a.headless:cv2.destroyAllWindows()
 print(json.dumps(dict(analyses=processed,last_state=r['state'] if r else None)))
if __name__=='__main__':main()
