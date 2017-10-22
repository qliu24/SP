from config import *
from sklearn.cluster import SpectralClustering

def learn_mix_model(category, eps=1e-7):
    vc_file = os.path.join(VCencode_dir, 'SP_{}_encoding_{}.pickle'.format(VC['layer'], category))
    with open(vc_file,'rb') as fh:
        all_info = pickle.load(fh)

    N = len(all_info)
    print('Total image number of {}: {}'.format(category, N))
    sp_num = len(all_info[0])

    feature_len = VC['num']*(2*SP['patch_r']+1)**2
    sp_models = [[None for kk in range(SP['cls_num'])] for pp in range(sp_num-2)]
    
    for pp in range(sp_num-2):
    # for pp in range(1):
        print('Learning mix model for SP{}...'.format(pp))
        
        # spectral clustering
        sim_file = os.path.join(Feat['cache_dir'],'simmat','simmat_{}_SP{}.pickle'.format(category,pp))
        with open(sim_file,'rb') as fh:
            mat_dis = pickle.load(fh)
            
        inst_num = mat_dis.shape[0]
        print('    total number of instances: {}'.format(inst_num))
        mat_full = mat_dis + mat_dis.T - np.ones((inst_num,inst_num))
        np.fill_diagonal(mat_full, 0)
        
        W_mat = 1. - mat_full
        print('    W_mat stats: {}, {}'.format(np.mean(W_mat), np.std(W_mat)))
        K1 = 2
        cls_solver = SpectralClustering(n_clusters=K1,affinity='precomputed', random_state=666)
        lb = cls_solver.fit_predict(W_mat)
        
        K2=2
        idx2 = []
        W_mat2 = []
        lb2 = []
        for k in range(K1):
            idx2.append(np.where(lb==k)[0])
            W_mat2.append(W_mat[np.ix_(idx2[k],idx2[k])])
            print('    W_mat_i stats: {}, {}'.format(np.mean(W_mat2[k]), np.std(W_mat2[k])))

            cls_solver = SpectralClustering(n_clusters=K2,affinity='precomputed', random_state=666)
            lb2.append(cls_solver.fit_predict(W_mat2[k]))

        rst_lbs1 = np.ones(len(idx2[0]))*-1
        rst_lbs1[np.where(lb2[0]==0)[0]] = 0
        rst_lbs1[np.where(lb2[0]==1)[0]] = 1
        rst_lbs2 = np.ones(len(idx2[1]))*-1
        rst_lbs2[np.where(lb2[1]==0)[0]] = 2
        rst_lbs2[np.where(lb2[1]==1)[0]] = 3
        rst_lbs = np.ones(inst_num)*-1
        rst_lbs[idx2[0]] = rst_lbs1
        rst_lbs[idx2[1]] = rst_lbs2
        rst_lbs = rst_lbs.astype('int')
        
        assert(SP['cls_num'] == K1*K2)
        for kk in range(SP['cls_num']):
            print('    cluster {} has {} samples'.format(kk, np.sum(rst_lbs==kk)))
        
        # compute factorizable model
        sp_fires = np.zeros((SP['cls_num'],feature_len))
        sp_cnt = np.zeros(SP['cls_num'])
        inst_idx = 0
        for nn in range(N):
            sp_pp_cnt = len(all_info[nn][pp])
            for mm in range(sp_pp_cnt):
                lb_mm = rst_lbs[inst_idx]
                sp_cnt[lb_mm] += 1
                sp_fires[lb_mm] += all_info[nn][pp][mm].astype(float)
                inst_idx += 1
                
        assert(inst_idx == inst_num)
        assert(np.sum(sp_cnt) == inst_num)
        for kk in range(SP['cls_num']):
            print('    cluster {} has {} samples'.format(kk, sp_cnt[kk]))
            freq = sp_fires[kk]/sp_cnt[kk]+eps
            weights = np.log(freq/(1-freq))
            logZ = np.sum(np.log(1.0/(1.0-freq)))
            logPrior = np.log(sp_cnt[kk]/inst_num)
            sp_models[pp][kk] = (weights, logZ, logPrior, np.log(inst_num))
            
            
    sp_fires_bg = [np.zeros(feature_len) for pp in range(2)]
    sp_cnt_bg = np.zeros(2)
    for nn in range(N):
        ppi = 0
        for pp in range(sp_num-2, sp_num):
            spi_cnt = len(all_info[nn][pp])
            sp_cnt_bg[ppi]+=spi_cnt

            for ii in range(spi_cnt):
                sp_fires_bg[ppi]+=(all_info[nn][pp][ii]).astype(float)
                
            ppi += 1
            
    print('Number of instances for BG models: {}'.format(sp_cnt_bg))
    
    for pp in range(2):
        freq = sp_fires_bg[pp].astype(float)/sp_cnt_bg[pp]+eps
        weights = np.log(freq/(1-freq))
        Z = np.sum(np.log(1.0/(1.0-freq)))
        prior = np.log(sp_cnt[pp])
        sp_models.append((weights, Z, prior))
    
    
    model_file = os.path.join(Model_dir, 'Mix_{}_{}.pickle'.format(VC['layer'], category))
    with open(model_file,'wb') as fh:
        pickle.dump(sp_models, fh)
        

if __name__=='__main__':
    for category in all_categories2[3:4]:
        learn_mix_model(category)