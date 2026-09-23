"""Safe NON-DOCUMENT proxy; no valid MRZ, identity, seal, or real passport image."""
from pathlib import Path
import json
import cv2
import numpy as np
from .geometry import box_quad
from .localization import Detection

GUIDE=dict(x=.16,y=.18,w=.68,h=.64)

def proxy():
    rng=np.random.default_rng(42)
    im=np.full((400,640,3),(201,217,224),np.uint8)
    noise=rng.normal(0,2,im.shape[:2])
    im=np.clip(im.astype(float)+noise[:,:,None],0,255).astype(np.uint8)
    cv2.rectangle(im,(3,3),(636,396),(65,90,95),3)
    cv2.putText(im,'SYNTHETIC QUALITY TEST - NOT A DOCUMENT',(22,34),cv2.FONT_HERSHEY_SIMPLEX,.60,(40,40,40),1,cv2.LINE_AA)
    cv2.rectangle(im,(26,72),(174,292),(142,170,180),-1)
    cv2.circle(im,(100,144),40,(97,121,126),-1)
    cv2.ellipse(im,(100,260),(60,65),0,180,360,(97,121,126),-1)
    for i,s in enumerate(['TEST PAGE ONLY','NO PERSONAL INFORMATION','EXAMPLE FIELD A: XXXX','EXAMPLE FIELD B: XXXX','INVALID SAMPLE 0000','DO NOT USE AS IDENTIFICATION']):
        cv2.putText(im,s,(196,90+i*36),cv2.FONT_HERSHEY_SIMPLEX,.51,(48,48,48),1,cv2.LINE_AA)
    for y in [343,378]:
        cv2.putText(im,'TEST<<INVALID<<SYNTHETIC<<NOT<<MRZ<<0000',(22,y),cv2.FONT_HERSHEY_SIMPLEX,.67,(30,30,30),2,cv2.LINE_AA)
    return im

def sample(kind='clean',level=0):
    card=proxy(); p=box_quad([192,184,832,584]); mrz=box_quad([20,316,620,390]); alpha=np.zeros(card.shape[:2],np.float32)
    if kind=='gaussian' and level>0: card=cv2.GaussianBlur(card,(0,0),level)
    elif kind=='motion' and level>1:
        size=int(level)|1; kernel=np.zeros((size,size),np.float32); kernel[size//2,:]=1/size; card=cv2.filter2D(card,-1,kernel)
    elif kind=='exposure': card=np.clip(card.astype(float)*level,0,255).astype(np.uint8)
    elif kind in ('glare','colored_glare') and level>0:
        yy,xx=np.mgrid[:400,:640]; radius=level
        alpha=np.clip((1-((xx-420)/(radius*1.7))**2-((yy-345)/radius)**2)*3,0,1).astype(np.float32)
        color=np.array([255,255,255] if kind=='glare' else [110,255,255])
        card=np.clip(card*(1-alpha[:,:,None])+color*alpha[:,:,None],0,255).astype(np.uint8)
    elif kind=='scale': p=(p-p.mean(0))*level+p.mean(0)
    elif kind=='position': p[:,0]+=level*1024*GUIDE['w']
    elif kind=='position_y': p[:,1]+=level*768*GUIDE['h']
    elif kind=='rotation':
        rad=np.deg2rad(level); rot=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
        p=(p-p.mean(0))@rot.T+p.mean(0)
    elif kind=='perspective': p[0,0]+=level*180; p[1,0]-=level*180; p[0,1]+=level*50
    p=p.astype(np.float32); H=cv2.getPerspectiveTransform(box_quad([0,0,639,399]),p)
    canvas=np.full((768,1024,3),45,np.uint8)
    warped=cv2.warpPerspective(card,H,(1024,768)); valid=cv2.warpPerspective(np.full((400,640),255,np.uint8),H,(1024,768))
    canvas[valid>0]=warped[valid>0]
    mp=cv2.perspectiveTransform(mrz[None],H)[0]
    gt=cv2.warpPerspective((alpha>.5).astype(np.uint8)*255,H,(1024,768))
    return canvas,Detection(p,mp,1.,'synthetic_ground_truth',True,1.),gt


def generate(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    groups={'clean':[0],'gaussian':[0,1,2,3,5,7],'motion':[1,5,11,19,31],
      'exposure':[.15,.25,.4,.6,.8,1.,1.15,1.3,1.6,2.],
      'scale':[.35,.5,.7,1.,1.2,1.5,1.8],'position':[-.4,-.2,0,.2,.4],
      'position_y':[-.4,-.2,0,.2,.4], 'rotation':[0,5,10,20,45,90,180,270],
      'perspective':[0,.2,.4,.6,.8,1.],'glare':[0,15,30,50,75], 'colored_glare':[0,15,30,50,75]}
    manifest=[]
    for kind,levels in groups.items():
        for i,level in enumerate(levels):
            im,det,gt=sample(kind,level); stem=f'{kind}_{i:02d}'
            cv2.imwrite(str(folder/(stem+'.png')),im); cv2.imwrite(str(folder/(stem+'_glare_gt.png')),gt)
            meta=dict(id=stem,image=stem+'.png',kind=kind,level=level,guide_box=GUIDE,detection=det.as_dict(),glare_gt=stem+'_glare_gt.png')
            (folder/(stem+'.json')).write_text(json.dumps(meta,indent=2)); manifest.append(meta)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
