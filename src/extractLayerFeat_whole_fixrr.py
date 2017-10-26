from scipy.spatial.distance import cdist
import scipy.io as sio
from FeatureExtractor import *
from config import *
import h5py

def extractLayerFeat_whole_fixrr(category, extractor, resize_tar, set_type='train'):
    print('extracting {} set features for {} at resize value {}'.format(set_type, category, resize_tar))
    # img_dir = Dataset['occ_img_dir'].format(category,'NINE')
    if set_type == 'occ':
        img_dir = Dataset['occ_img_dir'].format(category, SP['occ_level'])
        anno_dir = Dataset['anno_dir'].format(category)
        filelist = Dataset['{}_list'.format(set_type)].format(category)
        with open(filelist, 'r') as fh:
            contents = fh.readlines()
            
        img_list = [cc.strip()[0:-2] for cc in contents if cc != '\n']
        idx_list = [cc.strip()[-1] for cc in contents if cc != '\n']
        N = len(img_list)
        print('Total image number for {} set of {}: {}'.format(set_type, category, N))
        
    else:
        img_dir = Dataset['img_dir_org'].format(category)
        anno_dir = Dataset['anno_dir'].format(category)
        filelist = Dataset['{}_list'.format(set_type)].format(category)
        with open(filelist, 'r') as fh:
            contents = fh.readlines()

        img_list = [cc.strip().split()[0] for cc in contents if cc != '\n']
        idx_list = [cc.strip().split()[1] for cc in contents if cc != '\n']
        N = len(img_list)
        print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
    feat_set = [None for nn in range(N)]
    for nn in range(N):
        if nn%100==0:
            print(nn, end=' ', flush=True)
            
        if set_type == 'occ':
            matfile = os.path.join(img_dir, '{}_{}.mat'.format(img_list[nn], idx_list[nn]))
            f = h5py.File(matfile)
            img = np.array(f['record']['img']).T
            img = img[:,:,::-1]  # RGB to BGR
            
        else:

            img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
            try:
                assert(os.path.exists(img_file))
            except:
                print('file not exist: {}'.format(img_file))
                continue

            img = cv2.imread(img_file)
            
        resize_ratio = resize_tar/np.min(img.shape[0:2])
        img_resized = cv2.resize(img,None,fx=resize_ratio, fy=resize_ratio)
        
        layer_feature = extractor.extract_feature_image(img_resized)[0]
        assert(featDim == layer_feature.shape[2])
        feat_set[nn] = layer_feature
        
    print('\n')
    
    dir_feat_cache = os.path.join(Feat['cache_dir'], 'resize_{}'.format(resize_tar))
    if not os.path.exists(dir_feat_cache):
        os.makedirs(dir_feat_cache)
    
    file_cache_feat = os.path.join(dir_feat_cache, 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    with open(file_cache_feat, 'wb') as fh:
        pickle.dump(feat_set, fh)
        
    # file_cache_rr = os.path.join(Feat['cache_dir'], 'feat_{}_{}_rr.pickle'.format(category, set_type))
    # with open(file_cache_rr, 'wb') as fh:
    #     pickle.dump(resize_ratio_ls, fh)
        
            
if __name__=='__main__':
    resize_tar = int(sys.argv[1])
    extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
    for category in all_categories2[3:4]:
        extractLayerFeat_whole_fixrr(category, extractor, resize_tar, set_type='test')