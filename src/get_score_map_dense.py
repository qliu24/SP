from config import *

category = 'car'

def get_score_map_dense(category, set_type, dir_feat_cache, dir_save_scores):
    # load features
    file_cache_feat = os.path.join(dir_feat_cache, 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']+'_dense'))

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
        
    
    save_file = os.path.join(dir_save_scores, 'scores_{}_{}_{}.pickle'.format(category, set_type, VC['layer']+'_dense'))
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