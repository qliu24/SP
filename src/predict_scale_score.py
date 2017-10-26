from config import *
from scipy.spatial.distance import cdist
import scipy.io as sio
from FeatureExtractor import *
from ScoreComputer import ScoreComputer

def predict_scale(category, extractor, set_type):
    img_dir = Dataset['img_dir_org'].format(category)
    anno_dir = Dataset['anno_dir'].format(category)
    filelist = Dataset['{}_list'.format(set_type)].format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    img_list = [cc.strip().split()[0] for cc in contents if cc != '\n']
    idx_list = [cc.strip().split()[1] for cc in contents if cc != '\n']
    N = len(img_list)
    print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
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
    
    sc = ScoreComputer(SP['patch_r'], VC['num'], sp_num)
    scale_record = [None for nn in range(N)]
    score_raw_record = [None for nn in range(N)]
    for nn in range(N):
        if nn%100==0:
            print(nn)
            
        img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
        img = cv2.imread(img_file)
        scores = []
        for resize_tar in SP['resize_ls']:
            # extract layer feature
            resize_ratio = resize_tar/np.min(img.shape[0:2])
            img_resized = cv2.resize(img,None,fx=resize_ratio, fy=resize_ratio)
            lff = extractor.extract_feature_image(img_resized)[0]
            iheight,iwidth = lff.shape[0:2]
            lff = lff.reshape(-1, featDim)
            lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
            lfr = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
            lfb = (lfr<magic_thrh).astype(int)
            score_raw = sc.comptScore_mixture(lfb, sp_models, SP['cls_num'])
            score_final = [np.max(score_raw[:,:,pp] - np.max(score_raw[:,:,[-2,-1]], axis=2)) for pp in range(sp_num-2)]
            scores.append(np.copy(score_final))
            
        scale_vote = np.argmax(scores, axis=0)
        # final_scale = np.mean([SP['resize_ls'][svi] for svi in scale_vote])
        scale_most = np.argmax(np.bincount(scale_vote))
        final_scale = SP['resize_ls'][scale_most]
        scale_record[nn] = final_scale
        # print('predicted scale is {}'.format(final_scale))
        
        resize_ratio = final_scale/np.min(img.shape[0:2])
        img_resized = cv2.resize(img,None,fx=resize_ratio, fy=resize_ratio)
        lff = extractor.extract_feature_image(img_resized)[0]
        iheight,iwidth = lff.shape[0:2]
        lff = lff.reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        lfr = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
        lfb = (lfr<magic_thrh).astype(int)
        score_raw = sc.comptScore_mixture(lfb, sp_models, SP['cls_num'])
        score_raw_record[nn] = score_raw
        
    scale_save_file = os.path.join(Score_dir, 'scale_{}_{}.pickle'.format(category, set_type))
    with open(scale_save_file,'wb') as fh:
        pickle.dump(scale_record, fh)
        
    score_save_file = os.path.join(Score_dir, 'score_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    with open(score_save_file,'wb') as fh:
        pickle.dump(score_raw_record, fh)
        
        
if __name__=='__main__':
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    for category in all_categories2[3:4]:
        predict_scale(category, extractor, set_type='test')