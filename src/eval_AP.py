from config import *
from nms import nms
from VOCap import VOCap
from scipy.spatial.distance import cdist
import scipy.io as sio
from comptScores import comptScores
import h5py
from ScoreComputer import ScoreComputer


category = 'car'
set_type = 'test'
tf_flag = 'True'

# load model
# model should be a list with length = sp_num
# list elements are tuples with weight (feature_len 1D array), logZ (scalar), logPrior (scalar)
model_file = os.path.join(Model_dir, 'Unary_{}_{}.pickle'.format(VC['layer'], category))
with open(model_file,'rb') as fh:
    sp_models = pickle.load(fh)

sp_num = len(sp_models)


# model_file = '/mnt/1TB_SSD/qing/SP/mat_file/model_65.mat'
# matcontent = sio.loadmat(model_file)
# sp_num = matcontent['prior'].shape[1]
# sp_models = [None for pp in range(sp_num)]
# for pp in range(sp_num):
#     pp_prior = matcontent['prior'][0,pp]
#     pp_weight = matcontent['weight'][pp].T.ravel()
#     pp_logZ = np.sum(matcontent['logZ'][pp])
#     sp_models[pp] = (pp_weight, pp_logZ, pp_prior)
    

feature_len = VC['num']*(2*SP['patch_r']+1)**2
assert(len(sp_models[0][0])==feature_len)

# load VC centers
with open(Dict['Dictionary'].format(category),'rb') as fh:
    centers = pickle.load(fh)

# load magic threshold
thrh_file = os.path.join(Model_dir,'magic_thh_{}_{}.pickle'.format('train',VC['layer']))
with open(thrh_file, 'rb') as fh:
    thrh_ls = pickle.load(fh)

magic_thrh = thrh_ls[all_categories2.index(category)]

# magic_thrh = 0.65

# res_info_file = '/mnt/1TB_SSD/qing/SP/mat_file/res_info_test.mat'
# f = h5py.File(res_info_file)
# res_info = f['res_info']
# layer_feature_b = [None for nn in range(N)]
# for nn in range(N):
#     group = f[res_info[nn][0]]
#     layer_feature_b[nn] = (np.array(group['layer_feature_dist']).T<magic_thrh).astype(int)

# load features
file_cache_feat = os.path.join(Feat['cache_dir'], 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
with open(file_cache_feat, 'rb') as fh:
    layer_feature = pickle.load(fh)
    
N = len(layer_feature)
# N = 200
print('Total image number for {} set of {}: {}'.format(set_type, category, N))

# get test file list
filelist = Dataset['{}_list'.format(set_type)].format(category)
with open(filelist, 'r') as fh:
    contents = fh.readlines()

if set_type == 'occ':
    img_list = [cc.strip()[0:-2] for cc in contents if cc != '\n'][0:N]
    idx_list = [cc.strip()[-1] for cc in contents if cc != '\n'][0:N]
else:
    img_list = [cc.strip().split()[0] for cc in contents if cc != '\n'][0:N]
    idx_list = [cc.strip().split()[1] for cc in contents if cc != '\n'][0:N]

assert(len(img_list)==N)

# load resize_ratio
file_cache_rr = os.path.join(Feat['cache_dir'], 'feat_{}_{}_rr.pickle'.format(category, set_type))
with open(file_cache_rr, 'rb') as fh:
    resize_ratio_ls = pickle.load(fh)
    
# load gt bbox
img_dir = Dataset['img_dir_org'].format(category)
img_size = np.zeros((N, 2))
spanno = [[None for pp in range(sp_num-2)] for nn in range(N)]
for nn in range(N):
    # get image size
    img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
    img = cv2.imread(img_file)
    img_h, img_w = resize_ratio_ls[nn]*np.array(img.shape[0:2])
    img_size[nn] = resize_ratio_ls[nn]*np.array(img.shape[0:2])
    
    anno_file = os.path.join(Dataset['sp_anno_dir'].format(category), '{}.mat'.format(img_list[nn]))
    matcontent = sio.loadmat(anno_file)
    assert(sp_num-2 == matcontent['anno'][int(idx_list[nn])-1,1].shape[0])
    for pp in range(sp_num-2):
        sn_check = matcontent['anno'][int(idx_list[nn])-1,1][pp,0].shape[1]
        if sn_check == 0:
            spanno[nn][pp] = np.array([])
        elif sn_check>0 and sn_check!=9:
            print(nn, pp, matcontent['anno'][int(idx_list[nn])-1,1][pp,0].shape)
            spanno[nn][pp] = np.array([])
        else:
            # spanno[nn][pp] = matcontent['anno'][int(idx_list[nn])-1,1][pp,0]
            # process the annotations...
            spi_num = matcontent['anno'][int(idx_list[nn])-1,1][pp,0].shape[0]
            
            # spanno[nn][pp] = np.zeros((spi_num, 4))
            # for ii in range(spi_num):
            #     bb_o = matcontent['anno'][int(idx_list[nn])-1,1][pp,0][ii,0:4]
            #     bb_o = np.array([max(np.ceil(bb_o[0]),1), max(np.ceil(bb_o[1]),1), \
            #             min(np.floor(bb_o[2]), img_w),  min(np.floor(bb_o[3]), img_h)])
            #     spanno[nn][pp][ii] = bb_o
            
            # for resized whole image
            spanno[nn][pp] = np.zeros((spi_num, 4))
            for ii in range(spi_num):
                bb_o = matcontent['anno'][int(idx_list[nn])-1,1][pp,0][ii,4:8]*resize_ratio_ls[nn]
                bb_o = np.array([max(np.ceil(bb_o[0]),1), max(np.ceil(bb_o[1]),1), \
                        min(np.floor(bb_o[2]), img_w),  min(np.floor(bb_o[3]), img_h)])
                spanno[nn][pp][ii] = bb_o
                
                
# get scores for each pixel
if tf_flag:
    sc = ScoreComputer(SP['patch_r'], VC['num'], sp_num)

score_map = [None for nn in range(N)]
for nn in range(N):
    if nn%100==0:
        print(nn)
    
    # transfer features into VC 0/1
    iheight,iwidth = layer_feature[nn].shape[0:2]
    # iheight,iwidth = layer_feature_b[nn].shape[0:2]
    lff = layer_feature[nn].reshape(-1, featDim)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    lfr = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
    lfb = (lfr<magic_thrh).astype(int)
    # lfb = layer_feature_b[nn]
    
    lfb_padded = np.pad(lfb, ((SP['patch_r'],SP['patch_r']),(SP['patch_r'],SP['patch_r']),(0,0)), 'constant')
    if tf_flag:
        score_map[nn] = sc.comptScore(lfb, sp_models)
    else:
        score_map[nn] = np.zeros((iheight,iwidth, sp_num))
        for hh in range(iheight):
            for ww in range(iwidth):
                patch_feat = lfb_padded[hh:hh+2*SP['patch_r']+1, ww:ww+2*SP['patch_r']+1,:].ravel()
                score_map[nn][hh,ww] = comptScores(patch_feat, sp_models)
                
        
# Evaluation
ap_ls = []
# for pp in range(sp_num-2):
for pp in range(1):
    sp_detection = np.zeros((0,6))
    _t1 = time.time()
    for nn in range(N):
        det = score_map[nn][:,:,pp] - np.max(score_map[nn][:,:,np.arange(sp_num)!=pp], axis=2)
        
        [r_list, c_list] = np.unravel_index(range(det.shape[0]*det.shape[1]), (det.shape[0], det.shape[1]))
        # image level
        r_list = 16.0 * r_list + 7.5 + 1.0
        c_list = 16.0 * c_list + 7.5 + 1.0
        det = det.ravel()
        
        # nms_thresh = np.min(det) + (np.max(det) - np.min(det)) * SP['NMS_score_ratio']
        nms_thresh = np.median(det)
        r_list = r_list[det >= nms_thresh]
        c_list = c_list[det >= nms_thresh]
        det = det[det >= nms_thresh]
        
        bb_loc = np.column_stack((c_list-49.5, r_list-49.5, c_list+49.5, r_list+49.5, det))
        
        nms_list = nms(bb_loc, 0.15)
        bb_loc_ = np.concatenate((np.ones((len(nms_list), 1))*nn, bb_loc[nms_list]), axis=1)
        sp_detection = np.concatenate((sp_detection, bb_loc_), axis=0)
        
    _t2 = time.time()
    print('NN for loop: {}'.format(_t2-_t1))
    
    kp_pos = np.sum([spanno[nn][pp].shape[0] for nn in range(N)])
    
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
        if SP['criteria'] == 'dist':
            min_dist = np.inf
            inst = spanno[img_id][pp]
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
            inst = spanno[img_id][pp]
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
            
    _t3 = time.time()
    print('tp fp pass: {}'.format(_t3-_t2))
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/kp_pos
    prec = tp/(tp+fp)
    
    ap = VOCap(rec[9:], prec[9:])
    _t4 = time.time()
    print('mAP pass: {}'.format(_t4-_t3))
    
    print('SP{}, {:3.1f}'.format(pp, ap*100))
    ap_ls.append(ap)
    
print(np.mean(ap_ls))
            


