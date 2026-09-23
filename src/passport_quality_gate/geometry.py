"""All points are pixels in the supplied, already EXIF-oriented frame."""
import cv2
import numpy as np


def clip(x):
    return float(np.clip(x, 0, 1))


def order_quad(points):
    p = np.asarray(points, np.float32).reshape(4, 2)
    if not np.isfinite(p).all() or len(np.unique(p, axis=0)) != 4:
        raise ValueError("Quad requires four distinct finite points")
    # Angular sort avoids duplicate corners at 45 degrees (sum/difference tie).
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    if not cv2.isContourConvex(p) or abs(cv2.contourArea(p)) < 4:
        raise ValueError("Quad must be convex and non-degenerate")
    return p


def box_quad(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)


def guide_polygon(guide, shape):
    h,w = shape[:2]
    if isinstance(guide, dict):
        x,y,gw,gh = (float(guide[k]) for k in ('x','y','w','h'))
        if not all(np.isfinite([x,y,gw,gh])) or min(x,y)<0 or min(gw,gh)<=0 or x+gw>1 or y+gh>1:
            raise ValueError("guide_box must be a positive normalized rectangle inside [0,1]")
        return box_quad([x*w,y*h,(x+gw)*w,(y+gh)*h])
    p = order_quad(guide)
    if (p<0).any() or (p[:,0]>w).any() or (p[:,1]>h).any():
        raise ValueError("guide_polygon must be pixel coordinates inside frame")
    return p


def overlap(a,b):
    return max(0., float(cv2.intersectConvexConvex(np.asarray(a,np.float32), np.asarray(b,np.float32))[0]))


def analyze_geometry(det, guide, shape, cfg):
    p = order_quad(det.polygon)
    area = abs(cv2.contourArea(p)); ga=abs(cv2.contourArea(guide))
    extent=np.ptp(p,axis=0); ge=np.ptp(guide,axis=0)
    offset=(p.mean(0)-guide.mean(0))/ge
    edges=np.roll(p,-1,axis=0)-p; lengths=np.linalg.norm(edges,axis=1)
    ratios=[float(max(lengths[i],lengths[i+2])/max(1.,min(lengths[i],lengths[i+2]))) for i in (0,1)]
    angles=[]
    for i in range(4):
        a=p[i-1]-p[i]; b=p[(i+1)%4]-p[i]
        angles.append(float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)), -1,1)))))
    perspective=max(max(ratios)-1, max(abs(np.array(angles)-90))/90)
    angle=float(np.degrees(np.arctan2(edges[0,1],edges[0,0])))
    semantic=None
    if det.mrz_polygon is not None:
        delta=np.asarray(det.mrz_polygon).mean(0)-p.mean(0)
        # MRZ center should be below the page center in upright capture.
        semantic=float(np.degrees(np.arctan2(-delta[0],delta[1])))
        angle=semantic if abs(semantic)>45 else angle
    h,w=shape[:2]
    margin=float(min(p[:,0].min()/w,p[:,1].min()/h,(w-p[:,0].max())/w,(h-p[:,1].max())/h))
    containment=overlap(p,guide)/area
    frame_inside=overlap(p,box_quad([0,0,w,h]))/area
    pmin,pmax=p.min(0),p.max(0)
    mrz_inside=None
    mrz_width_ratio=None
    mrz_aspect_ratio=None
    mrz_center_x_rel=None
    mrz_center_y_rel=None
    if det.mrz_polygon is not None:
        mq=order_quad(det.mrz_polygon)
        mrz_inside=overlap(p,mq)/max(1,abs(cv2.contourArea(mq)))
        mmin,mmax=mq.min(0),mq.max(0)
        mext=np.maximum(1.,mmax-mmin)
        mrz_width_ratio=float(mext[0]/max(1.,extent[0]))
        mrz_aspect_ratio=float(mext[0]/max(1.,mext[1]))
        mcenter=(mmin+mmax)/2
        mrz_center_x_rel=float((mcenter[0]-pmin[0])/max(1.,extent[0]))
        mrz_center_y_rel=float((mcenter[1]-pmin[1])/max(1.,extent[1]))
    crop_risk=max(clip(1-margin/cfg['border_margin']),clip((1-frame_inside)*5),float(mrz_inside is not None and mrz_inside<0.95))
    # Signed edge gaps relative to the guide. Positive means the page edge
    # remains inside the guide; negative means it extends past that guide edge.
    gmin,gmax=guide.min(0),guide.max(0)
    edge_gaps=dict(
        left=float((pmin[0]-gmin[0])/max(1.,ge[0])),
        right=float((gmax[0]-pmax[0])/max(1.,ge[0])),
        top=float((pmin[1]-gmin[1])/max(1.,ge[1])),
        bottom=float((gmax[1]-pmax[1])/max(1.,ge[1])),
    )
    # Physical camera-edge gaps are different from guide gaps.  They are used
    # for directional recovery when only part of the document is visible.
    frame_edge_gaps=dict(
        left=float(pmin[0]/max(1.,w)),
        right=float((w-pmax[0])/max(1.,w)),
        top=float(pmin[1]/max(1.,h)),
        bottom=float((h-pmax[1])/max(1.,h)),
    )
    page_aspect_ratio=float(extent[0]/max(1.,extent[1]))
    return dict(fill_ratio=area/ga,fill_w=float(extent[0]/ge[0]),fill_h=float(extent[1]/ge[1]),
        center_offset_x=float(offset[0]),center_offset_y=float(offset[1]),edge_gaps=edge_gaps,
        frame_edge_gaps=frame_edge_gaps,page_aspect_ratio=page_aspect_ratio,
        mrz_width_ratio=mrz_width_ratio,mrz_aspect_ratio=mrz_aspect_ratio,
        mrz_center_x_rel=mrz_center_x_rel,mrz_center_y_rel=mrz_center_y_rel,rotation_deg=angle,
        semantic_rotation_deg=semantic,orientation_known=semantic is not None,
        perspective_score=float(perspective) if det.corners_reliable else None,
        opposite_edge_ratios=ratios,corner_angles_deg=angles,containment=containment,
        frame_containment=frame_inside,border_margin=margin,crop_risk=crop_risk,
        corners_reliable=det.corners_reliable,mrz_inside=mrz_inside)


def rectify(frame, det, width):
    p=order_quad(det.polygon)
    if det.mrz_polygon is not None:
        midpoints=(p+np.roll(p,-1,axis=0))/2
        edge=int(np.argmin(np.linalg.norm(midpoints-np.asarray(det.mrz_polygon).mean(0),axis=1)))
        p=np.roll(p,2-edge,axis=0) # Place the MRZ-adjacent edge at the bottom before scaling.
    lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    height=max(16,int(round(width*(lengths[1]+lengths[3])/(lengths[0]+lengths[2]))))
    height=min(height,width*3)
    matrix=cv2.getPerspectiveTransform(p,box_quad([0,0,width-1,height-1]))
    crop=cv2.warpPerspective(frame,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # Validity mask excludes any out-of-frame pixels from quality statistics.
    valid=cv2.warpPerspective(np.full(frame.shape[:2],255,np.uint8),matrix,(width,height),flags=cv2.INTER_NEAREST)
    mrz=np.zeros((height,width),np.uint8)
    if det.mrz_polygon is not None:
        q=cv2.perspectiveTransform(np.asarray(det.mrz_polygon,np.float32)[None],matrix)[0]
        cv2.fillConvexPoly(mrz,np.round(q).astype(np.int32),255)
        mrz=cv2.bitwise_and(mrz,valid)
    return crop,valid,mrz,matrix
