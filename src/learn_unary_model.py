from config import *

def learn_unary_model(category, eps=1e-7):
    vc_file = os.path.join(VCencode_dir, 'SP_{}_encoding_{}.pickle'.format(VC['layer'], category))
    with open(vc_file,'rb') as fh:
        all_info = pickle.load(fh)

    N = len(all_info)
    print('Total image number of {}: {}'.format(category, N))
    sp_num = len(all_info[0])
    assert(np.all([len(ff)==sp_num for ff in all_info])) # all have the same sp num

    feature_len = VC['num']*(2*SP['patch_r']+1)**2
    sp_fires = [np.zeros(feature_len) for pp in range(sp_num)]
    sp_cnt = np.zeros(sp_num)

    for nn in range(N):
        if nn%100==0:
            print(nn)

        for pp in range(sp_num):
            spi_cnt = len(all_info[nn][pp])
            sp_cnt[pp]+=spi_cnt

            for ii in range(spi_cnt):
                sp_fires[pp]+=(all_info[nn][pp][ii]).astype(float)

    print(sp_cnt)

    sp_models = [None for pp in range(sp_num)]
    for pp in range(sp_num):
        freq = sp_fires[pp].astype(float)/sp_cnt[pp]+eps
        weights = np.log(freq/(1-freq))
        Z = np.sum(np.log(1.0/(1.0-freq)))
        prior = np.log(sp_cnt[pp])
        sp_models[pp] = (weights, Z, prior)


    model_file = os.path.join(Model_dir, 'Unary_{}_{}.pickle'.format(VC['layer'], category))
    with open(model_file,'wb') as fh:
        pickle.dump(sp_models, fh)
        

if __name__=='__main__':
    for category in all_categories2[3:4]:
        learn_unary_model(category)