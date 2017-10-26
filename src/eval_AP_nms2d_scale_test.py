from config import *
import scipy.io as sio
import h5py
from joblib import Parallel, delayed
from eval_AP_inner import eval_AP_inner
from scipy.spatial.distance import cdist
from FeatureExtractor import *
from ScoreComputer import ScoreComputer

paral_num = 6
category = 'car'
set_type = 'test'

scale_save_file = os.path.join(Score_dir, 'scale_{}_{}.pickle'.format(category, set_type))
with open(scale_save_file,'rb') as fh:
    scale_record = pickle.load(fh)
    
# score_save_file = os.path.join(Score_dir, 'score_{}_{}_{}.pickle'.format(category, set_type, VC['layer']+'_dense'))
# with open(score_save_file,'rb') as fh:
#     score_map = pickle.load(fh)
    
sp_num = 41
N = len(scale_record)

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
    

# load gt bbox
img_dir = Dataset['img_dir_org'].format(category)
img_size = np.zeros((N, 2))
spanno = [[None for pp in range(sp_num-2)] for nn in range(N)]
for nn in range(N):
    # get image size
    img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
    img = cv2.imread(img_file)
    resize_ratio = scale_record[nn]/np.min(img.shape[0:2])
    img_h, img_w = resize_ratio*np.array(img.shape[0:2])
    img_size[nn] = resize_ratio*np.array(img.shape[0:2])
    
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
                bb_o = (matcontent['anno'][int(idx_list[nn])-1,1][pp,0][ii,4:8]-1)*resize_ratio+1
                bb_o = np.array([max(np.ceil(bb_o[0]),1), max(np.ceil(bb_o[1]),1), \
                        min(np.floor(bb_o[2]), img_w),  min(np.floor(bb_o[3]), img_h)])
                spanno[nn][pp][ii] = bb_o
            
            
# load VC centers
with open(Dict['Dictionary'].format(category),'rb') as fh:
    centers = pickle.load(fh)

# load magic threshold
thrh_file = os.path.join(Model_dir,'magic_thh_{}_{}.pickle'.format('train',VC['layer']))
with open(thrh_file, 'rb') as fh:
    thrh_ls = pickle.load(fh)

magic_thrh = thrh_ls[all_categories2.index(category)]

# Mix models
model_file = os.path.join(Model_dir, 'Mix_{}_{}.pickle'.format(VC['layer'], category))
with open(model_file,'rb') as fh:
    sp_models = pickle.load(fh)

sp_num = len(sp_models)

feature_len = VC['num']*(2*SP['patch_r']+1)**2
assert(len(sp_models[0][0][0])==feature_len)

# modify the weights so that it matches the dense features
for pp in range(sp_num-2):
    for kk in range(SP['cls_num']):
        weights_old = sp_models[pp][kk][0].reshape(2*SP['patch_r']+1, 2*SP['patch_r']+1, VC['num'])
        weights_new = np.zeros((4*SP['patch_r']+1, 4*SP['patch_r']+1, VC['num']))
        for wrr in range(0,4*SP['patch_r']+1,2):
            for wcc in range(0,4*SP['patch_r']+1,2):
                weights_new[wrr,wcc] = weights_old[wrr//2, wcc//2]
                
        sp_models[pp][kk] = (weights_new.ravel(),sp_models[pp][kk][1],sp_models[pp][kk][2],sp_models[pp][kk][3])
        
for pp in range(sp_num-2, sp_num):
    weights_old = sp_models[pp][0].reshape(2*SP['patch_r']+1, 2*SP['patch_r']+1, VC['num'])
    weights_new = np.zeros((4*SP['patch_r']+1, 4*SP['patch_r']+1, VC['num']))
    for wrr in range(0,4*SP['patch_r']+1,2):
        for wcc in range(0,4*SP['patch_r']+1,2):
            weights_new[wrr,wcc] = weights_old[wrr//2, wcc//2]

    sp_models[pp] = (weights_new.ravel(),sp_models[pp][1],sp_models[pp][2])
        
# get scores for each pixel
sc = ScoreComputer(SP['patch_r']*2, VC['num'], sp_num)
extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer']+'_dense', which_snapshot=0)
score_map = [None for nn in range(N)]
for nn in range(N):
    if nn%100==0:
        print(nn)
        
    img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
    img = cv2.imread(img_file)
    resize_ratio = scale_record[nn]/np.min(img.shape[0:2])
    img_resized = cv2.resize(img,None,fx=resize_ratio, fy=resize_ratio)
    lff = extractor.extract_feature_image(img_resized)[0]
    iheight,iwidth = lff.shape[0:2]
    lff = lff.reshape(-1, featDim)
    lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
    lfr = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
    lfb = (lfr<magic_thrh).astype(int)
    score_map[nn] = sc.comptScore_mixture(lfb, sp_models, SP['cls_num'])
    
    
# Evaluation
# rich BG
# inp_ls = [([score_map[nn][:,:,pp] - np.max(score_map[nn][:,:,np.arange(sp_num)!=pp], axis=2) for nn in range(N)], \
#            [spanno[nn][pp] for nn in range(N)], img_size, 2) for pp in range(sp_num-2)]

# simple BG
inp_ls = [([score_map[nn][:,:,pp] - np.max(score_map[nn][:,:,[-2,-1]], axis=2) for nn in range(N)], \
           [spanno[nn][pp] for nn in range(N)], img_size, 2) for pp in range(sp_num-2)]


ap_ls = np.array(Parallel(n_jobs=paral_num)(delayed(eval_AP_inner)(i) for i in inp_ls))
for pp in range(sp_num-2):
    print('SP{}, {:3.1f}'.format(pp, ap_ls[pp]*100))
    
print(np.mean(ap_ls))



            



