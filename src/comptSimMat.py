from joblib import Parallel, delayed
from vcdist_funcs import *
from config import *
import time

paral_num = 6

def comptSimMat(category):
    vc_file = os.path.join(VCencode_dir, 'SP_{}_encoding_{}.pickle'.format(VC['layer'], category))
    with open(vc_file,'rb') as fh:
        all_info = pickle.load(fh)

    N = len(all_info)
    print('Total image number of {}: {}'.format(category, N))
    sp_num = len(all_info[0])
    assert(np.all([len(ff)==sp_num for ff in all_info])) # all have the same sp num

    for pp in range(sp_num):
        print('computing similarity matrix for {} SP{}...'.format(category, pp))
        savename = os.path.join(Feat['cache_dir'],'simmat','simmat_{}_SP{}.pickle'.format(category,pp))

        sp_cnt = 0
        samples_pp = []
        for nn in range(N):
            sp_pp_cnt = len(all_info[nn][pp])
            sp_cnt+=sp_pp_cnt
            for mm in range(sp_pp_cnt):
                samples_pp.append(all_info[nn][pp][mm].reshape(2*SP['patch_r']+1,2*SP['patch_r']+1,VC['num']))

        assert(sp_cnt == len(samples_pp))
        print('    total sample number: {}'.format(sp_cnt))
        _s = time.time()
        inputs = [(samples_pp, ss) for ss in range(sp_cnt)]
        mat_dis = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
        _e = time.time()
        print('    total time: {}'.format((_e-_s)/60))

        with open(savename, 'wb') as fh:
            pickle.dump(mat_dis, fh)
        
        
if __name__=='__main__':
    for category in all_categories2[3:4]:
        comptSimMat(category)
