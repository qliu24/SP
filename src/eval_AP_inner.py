from config import *
from VOCap import VOCap
from DenseNMS import DenseNMS

def eval_AP_inner(inp):
    score_map, spanno, img_size, resolution = inp
    nms = DenseNMS(int(16 // resolution), 0.05)
    N = len(score_map)
    sp_detection = np.zeros((0,6))
    for nn in range(N):
        det = score_map[nn]
        rc_list = nms.nms(det)
        rc_list = np.array(rc_list)
        det = det[rc_list[:, 0], rc_list[:, 1]]
        r_list = 16.0 / resolution * rc_list[:, 0] + 16.0 / 2 - 0.5 + 1.0  # Matlab
        c_list = 16.0 / resolution * rc_list[:, 1] + 16.0 / 2 - 0.5 + 1.0
        bb_loc = np.column_stack((c_list - 49.5, r_list - 49.5, c_list + 49.5, r_list + 49.5, det))
        bb_loc_ = np.concatenate((np.ones((bb_loc.shape[0], 1)) * nn, bb_loc), axis=1)
        sp_detection = np.concatenate((sp_detection, bb_loc_), axis=0)
        
        
    kp_pos = np.sum([spanno[nn].shape[0] for nn in range(N)])
    
    tot = sp_detection.shape[0]
    sort_idx = np.argsort(-sp_detection[:,5])
    id_list = sp_detection[sort_idx, 0]
    col_list = (sp_detection[sort_idx, 1]+sp_detection[sort_idx, 3])/2
    row_list = (sp_detection[sort_idx, 2]+sp_detection[sort_idx, 4])/2
    bbox_list = sp_detection[sort_idx, 1:5].astype(int)
    
    tp = np.zeros(tot)
    fp = np.zeros(tot)
    flag = np.zeros((N,20))
    for dd in range(tot):
        if np.sum(flag)==kp_pos:
            fp[dd:]=1
            break
        
        img_id = int(id_list[dd])
        col_c = col_list[dd]
        row_c = row_list[dd]
        
        inst = spanno[img_id]
        if SP['criteria'] == 'dist':
            min_dist = np.inf
            for ii in range(inst.shape[0]):
                xx = (inst[ii,0]+inst[ii,2])/2
                yy = (inst[ii,1]+inst[ii,3])/2

                if np.sqrt((xx-col_c)**2+(yy-row_c)**2) < min_dist:
                    min_dist = np.sqrt((xx-col_c)**2+(yy-row_c)**2)
                    min_idx = ii

            if min_dist < SP['dist_thresh'] and flag[img_id, min_idx] == 0:
                tp[dd] = 1
                flag[img_id, min_idx]=1
            else:
                fp[dd] = 1

        elif SP['criteria'] == 'iou':
            max_iou = -np.inf
            for ii in range(inst.shape[0]):
                bbgt = inst[ii]
                bb = bbox_list[dd]
                bb = np.array([max(np.ceil(bb[0]),1), max(np.ceil(bb[1]),1), \
                               min(np.floor(bb[2]), img_size[img_id][1]),  min(np.floor(bb[3]), img_size[img_id][0])])
                
                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2]-bi[0]+1
                ih = bi[3]-bi[1]+1

                if iw>0 and ih>0:
                    ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+\
                         (bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-\
                         iw*ih
                    ov = iw*ih/ua
                    if ov>max_iou:
                        max_iou = ov
                        max_idx = ii

            if max_iou > SP['iou_thresh'] and flag[img_id, max_idx] == 0:
                tp[dd] = 1
                flag[img_id, max_idx]=1
            else:
                fp[dd] = 1
            
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/kp_pos
    prec = tp/(tp+fp)
    
    ap = VOCap(rec[9:], prec[9:])
    return ap