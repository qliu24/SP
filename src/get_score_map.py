from config import *
from ScoreComputer import ScoreComputer

def get_score_map(category, set_type, dir_feat_cache, dir_save_scores):
    # load features
    file_cache_feat = os.path.join(dir_feat_cache, 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))

    with open(file_cache_feat, 'rb') as fh:
        layer_feature = pickle.load(fh)

    N = len(layer_feature)
    # N = 200
    print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
    # load models
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
    
    # compute scores
    sc = ScoreComputer(SP['patch_r'], VC['num'], sp_num)
    score_map = [None for nn in range(N)]
    for nn in range(N):
        if nn%100==0:
            print(nn)

        # transfer features into VC 0/1
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        lfr = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
        lfb = (lfr<magic_thrh).astype(int)
        
        score_map[nn] = sc.comptScore_mixture(lfb, sp_models, SP['cls_num'])
        
    
    save_file = os.path.join(dir_save_scores, 'scores_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    with open(save_file, 'wb') as fh:
        pickle.dump(score_map, fh)
        
if __name__=='__main__':
    for resize_tar in SP['resize_ls']:
        dir_feat_cache = os.path.join(Feat['cache_dir'], 'resize_{}'.format(resize_tar))
        dir_save_scores = os.path.join(Score_dir, 'resize_{}'.format(resize_tar))
        if not os.path.exists(dir_save_scores):
            os.makedirs(dir_save_scores)
        
        for categroy in all_categories2:
            get_score_map_dense(category, 'test', dir_feat_cache, dir_save_scores)