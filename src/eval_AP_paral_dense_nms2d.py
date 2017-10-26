from config import *
from scipy.spatial.distance import cdist
import scipy.io as sio
import h5py
from joblib import Parallel, delayed
from ScoreComputer import ScoreComputer
from eval_AP_inner import eval_AP_inner

paral_num = 6
category = 'car'
set_type = 'test'


# load model
# model should be a list with length = sp_num
# list elements are tuples with weight (feature_len 1D array), logZ (scalar), logPrior (scalar)

# Unary models
# model_file = os.path.join(Model_dir, 'Unary_{}_{}.pickle'.format(VC['layer'], category))
# with open(model_file,'rb') as fh:
#     sp_models = pickle.load(fh)

# sp_num = len(sp_models)

# feature_len = VC['num']*(2*SP['patch_r']+1)**2
# assert(len(sp_models[0][0])==feature_len)

# Mix models
model_file = os.path.join(Model_dir, 'Mix_{}_{}.pickle'.format(VC['layer'], category))
with open(model_file,'rb') as fh:
    sp_models = pickle.load(fh)

sp_num = len(sp_models)

feature_len = VC['num']*(2*SP['patch_r']+1)**2
assert(len(sp_models[0][0][0])==feature_len)

# load VC centers
with open(Dict['Dictionary'].format(category),'rb') as fh:
    centers = pickle.load(fh)

# load magic threshold
thrh_file = os.path.join(Model_dir,'magic_thh_{}_{}.pickle'.format('train',VC['layer']))
with open(thrh_file, 'rb') as fh:
    thrh_ls = pickle.load(fh)

magic_thrh = thrh_ls[all_categories2.index(category)]

# load features
file_cache_feat = os.path.join(Feat['cache_dir'], 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']+'_dense'))
if set_type=='occ' and SP['occ_level'] != 'ONE':
    file_cache_feat = os.path.join(Feat['cache_dir'], \
                                   'feat_{}_{}_{}.pickle'.format(category, \
                                                                 '{}_{}'.format(set_type, SP['occ_level']), \
                                                                 VC['layer']+'_dense'))
    
with open(file_cache_feat, 'rb') as fh:
    layer_feature = pickle.load(fh)
    
N = len(layer_feature)
N = 200
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
                bb_o = (matcontent['anno'][int(idx_list[nn])-1,1][pp,0][ii,4:8]-1)*resize_ratio_ls[nn]+1
                bb_o = np.array([max(np.ceil(bb_o[0]),1), max(np.ceil(bb_o[1]),1), \
                        min(np.floor(bb_o[2]), img_w),  min(np.floor(bb_o[3]), img_h)])
                spanno[nn][pp][ii] = bb_o
            
            
# get scores for each pixel
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
    lfb_height, lfb_width = lfb.shape[0:2]
    
    rr_offset = [0,1]
    cc_offset = [0,1]
    lfb_ls = []
    msk_ls = []
    
    for rr_os in rr_offset:
        for cc_os in cc_offset:
            msk_base = np.zeros_like(lfb)
            for rr_msk in range(rr_os,lfb_height, 2):
                for cc_msk in range(cc_os, lfb_width, 2):
                    msk_base[rr_msk, cc_msk] = 1
                    
            msk_base = msk_base.astype(bool)
            
            lfb_height_new = np.sum(msk_base[:,cc_os,0])
            lfb_width_new = np.sum(msk_base[rr_os,:,0])
            lfb_selected = (lfb.ravel()[msk_base.ravel()]).reshape(lfb_height_new, lfb_width_new, -1)
            assert(lfb_selected.shape[2]==VC['num'])
            
            msk_ls.append(msk_base[:,:,0:sp_num].copy())
            lfb_ls.append(lfb_selected.copy())
            
    score_all = np.zeros((lfb_height, lfb_width, sp_num)).ravel()
    for msk_curr,lfb_curr in zip(msk_ls, lfb_ls):
        score_curr = sc.comptScore_mixture(lfb_curr, sp_models, SP['cls_num'])
        score_all[msk_curr.ravel()] = score_curr.ravel()
        
    score_all = score_all.reshape(lfb_height, lfb_width, -1)
    assert(score_all.shape[2]==sp_num)
    
    score_map[nn] = score_all.copy()
    
# Evaluation
# rich BG
# inp_ls = [([score_map[nn][:,:,pp] - np.max(score_map[nn][:,:,np.arange(sp_num)!=pp], axis=2) for nn in range(N)], \
#            [spanno[nn][pp] for nn in range(N)], img_size, 2) for pp in range(sp_num-2)]

# simple BG
inp_ls = [([score_map[nn][:,:,pp] - np.max(score_map[nn][:,:,[-2,-1]], axis=2) for nn in range(N)], \
           [spanno[nn][pp] for nn in range(N)], img_size, 2) for pp in range(sp_num-2)]


# debug mode
# with open('/mnt/1TB_SSD/qing/SP/debug.pickle', 'wb') as fh:
#     pickle.dump([score_map, spanno, img_size], fh)

ap_ls = np.array(Parallel(n_jobs=paral_num)(delayed(eval_AP_inner)(i) for i in inp_ls))
for pp in range(sp_num-2):
    print('SP{}, {:3.1f}'.format(pp, ap_ls[pp]*100))
    
print(np.mean(ap_ls))



            



