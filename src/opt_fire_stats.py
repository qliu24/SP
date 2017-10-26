from scipy.spatial.distance import cdist
from config import *
import scipy.io as sio

def opt_fire_stats(category, centers, set_type, empty_rate = 0.2):
    magic_thh_ls = [0.30]
    step = 0.05
    
    img_dir = Dataset['img_dir_org'].format(category)
    anno_dir = Dataset['anno_dir'].format(category)
    filelist = Dataset['{}_list'.format(set_type)].format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    img_list = [cc.strip().split()[0] for cc in contents if cc != '\n']
    idx_list = [cc.strip().split()[1] for cc in contents if cc != '\n']

    print('optimizing magic threshold for category {}, set_type {}, layer {}'.format(category, set_type, VC['layer']))
    filename = os.path.join(Feat['cache_dir'], 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    with open(filename, 'rb') as fh:
        layer_feature = pickle.load(fh)
    
    N = len(layer_feature)
    # print('{0}: total number of instances {1}'.format(category, N))
    # print(layer_feature[0].shape)
    
    file_cache_rr = os.path.join(Feat['cache_dir'], 'feat_{}_{}_rr.pickle'.format(category, set_type))
    with open(file_cache_rr, 'rb') as fh:
        resize_ratio_ls = pickle.load(fh)
    
    layer_feature_dist = []
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        lfd = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
        
        
        # select features that are inside bbox
        img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
        img = cv2.imread(img_file)
        img_h, img_w = img.shape[0:2]
        
        anno_file = os.path.join(anno_dir, '{}.mat'.format(img_list[nn]))
        matcontent = sio.loadmat(anno_file)
        bbox_value = matcontent['record']['objects'][0,0][0,int(idx_list[nn])-1]['bbox'][0]
        bbox_value = [max(math.ceil(bbox_value[0]), 1), max(math.ceil(bbox_value[1]), 1), \
                        min(math.floor(bbox_value[2]), img_w), min(math.floor(bbox_value[3]), img_h)]
        
        resize_ratio = resize_ratio_ls[nn]
        img_h_r = np.round(img_h*resize_ratio)
        img_w_r = np.round(img_w*resize_ratio)
        bbox_value_r = np.array(bbox_value)*resize_ratio
        bbox_value_r = [max(math.ceil(bbox_value_r[0]), 1), max(math.ceil(bbox_value_r[1]), 1), \
                        min(math.floor(bbox_value_r[2]), img_w_r), min(math.floor(bbox_value_r[3]), img_h_r)]
        bbox_value_r = np.array(bbox_value_r)
        pool_value_r = np.round((bbox_value_r - 1.0 - 7.5) / 16.0).astype(np.int32)
        lfd_cropped = lfd[pool_value_r[1]:pool_value_r[3], pool_value_r[0]:pool_value_r[2], :]
        # print(lfd_cropped.shape)
        
        layer_feature_dist.append(lfd_cropped)
        
        
    while step > 0.0005:
        magic_thh = magic_thh_ls[-1]
        layer_feature_b = [None for nn in range(N)]

        for nn in range(N):
            layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)

        vc_fire_cnt = [None for nn in range(N)]
        vc_fire_empty = [None for nn in range(N)]
        for nn in range(N):
            vc_fire_cnt[nn] = np.mean(np.sum(layer_feature_b[nn], axis=2))
            if np.isnan(np.mean(vc_fire_cnt[nn])):
                print(nn)
            
            vc_fire_empty[nn] = np.sum(np.sum(layer_feature_b[nn], axis=2)==0)/np.prod(layer_feature_b[nn].shape[0:2])

        
            
        print('Magic threshold {2}: {0}, {1}'.format(np.mean(vc_fire_cnt), np.mean(vc_fire_empty), magic_thh))
            
        if np.mean(vc_fire_empty) > empty_rate:
            magic_thh_ls.append(np.around(magic_thh+step, decimals=3))
        else:
            del(magic_thh_ls[-1])
            step /= 2.0
            magic_thh_ls.append(np.around(magic_thh_ls[-1]+step, decimals=3))
            
            
    magic_thh = magic_thh_ls[-1]
    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)

    vc_fire_cnt = [None for nn in range(N)]
    vc_fire_empty = [None for nn in range(N)]
    for nn in range(N):
        vc_fire_cnt[nn] = np.mean(np.sum(layer_feature_b[nn], axis=2))
        vc_fire_empty[nn] = np.sum(np.sum(layer_feature_b[nn], axis=2)==0)/np.prod(layer_feature_b[nn].shape[0:2])
        
    print('final rst: {}, {}, {}'.format(magic_thh, np.mean(vc_fire_cnt), np.mean(vc_fire_empty)))
    return(magic_thh)



if __name__=='__main__':
    set_type= 'train'

    rst_ls = []
    for category in all_categories2:
        with open(Dict['Dictionary'].format(category), 'rb') as fh:
            centers=pickle.load(fh)

        rst_ls.append(opt_fire_stats(category, centers, set_type))
        
        
    save_file = os.path.join(Model_dir,'magic_thh_{}_{}.pickle'.format(set_type, VC['layer']))
    
    with open(save_file, 'wb') as fh:
        pickle.dump(rst_ls, fh)

