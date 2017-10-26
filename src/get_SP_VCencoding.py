from scipy.spatial.distance import cdist
import scipy.io as sio
import matplotlib.pyplot as plt
from config import *
import h5py

def get_SP_VCencoding(category, set_type='train'):
    # get magic threshold
    thrh_file = os.path.join(Model_dir,'magic_thh_{}_{}.pickle'.format(set_type,VC['layer']))
    with open(thrh_file, 'rb') as fh:
        thrh_ls = pickle.load(fh)

    magic_thrh = thrh_ls[all_categories2.index(category)]

    # get file list
    filelist = Dataset['{}_list'.format(set_type)].format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    img_list = [cc.strip().split()[0] for cc in contents if cc != '\n']
    idx_list = [cc.strip().split()[1] for cc in contents if cc != '\n']

    N = len(img_list)
    print('Total image number for {} set of {}: {}'.format(set_type, category, N))

    # read in VC centers
    with open(Dict['Dictionary'].format(category),'rb') as fh:
        centers = pickle.load(fh)

    # get instance features and encode into 0/1
    file_feat = os.path.join(Feat['cache_dir'], 'feat_{}_{}_{}.pickle'.format(category, set_type, VC['layer']))
    with open(file_feat, 'rb') as fh:
        layer_feature = pickle.load(fh)

    assert(N == len(layer_feature))

    r_set = [None for nn in range(N)]
    for nn in range(N):
        iheight,iwidth = layer_feature[nn].shape[0:2]
        lff = layer_feature[nn].reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        r_set[nn] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)

    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = r_set[nn]<magic_thrh
        
        
    feature_len = VC['num']*(2*SP['patch_r']+1)**2
    all_info = [None for nn in range(N)]
    
    file_cache_rr = os.path.join(Feat['cache_dir'], 'feat_{}_{}_rr.pickle'.format(category, set_type))
    with open(file_cache_rr, 'rb') as fh:
        resize_ratio_ls = pickle.load(fh)

    for nn in range(N):
        if nn%100==0:
            print(nn)
            
        # get resize_ratio
        resize_ratio = resize_ratio_ls[nn]
        
        lfb = layer_feature_b[nn]
        height, width = lfb.shape[0:2]
        padded = np.pad(lfb, ((SP['patch_r'],SP['patch_r']),(SP['patch_r'],SP['patch_r']),(0,0)), 'constant').astype(bool)

        anno_file = os.path.join(Dataset['sp_anno_dir'].format(category), '{}.mat'.format(img_list[nn]))
        matcontent = sio.loadmat(anno_file)
        sp_num = matcontent['anno'][int(idx_list[nn])-1,1].shape[0]
        instance_info = [None for mm in range(sp_num+2)]

        cover_msk = np.zeros((height, width))

        for mm in range(sp_num):
            sn_check = matcontent['anno'][int(idx_list[nn])-1,1][mm,0].shape[1]
            if sn_check>0 and sn_check!=9:
                print(nn,mm, matcontent['anno'][int(idx_list[nn])-1,1][mm,0].shape)
                spi_num=0
            else:
                spi_num = matcontent['anno'][int(idx_list[nn])-1,1][mm,0].shape[0]

            sp_info = [None for kk in range(spi_num)]
            for kk in range(spi_num):
                spi_box = matcontent['anno'][int(idx_list[nn])-1,1][mm,0][kk]
                
                # for resized whole image:
                spi_box = (spi_box[4:8]-1)*resize_ratio+1

                xy = (spi_box[0:2]+spi_box[2:4])/2
                # pool_xy = (xy//Astride).astype(int)
                pool_xy = np.round((xy - 1.0 - 7.5) / 16.0).astype(np.int32) 

                if pool_xy[0] < 0:
                    # print(xy[0],pool_xy[0])
                    pool_xy[0] = 0
                if pool_xy[0] > width - 1:
                    # print(xy[0], pool_xy[0], width - 1)
                    pool_xy[0] = width - 1
                if pool_xy[1] < 0:
                    # print(xy[1],pool_xy[1])
                    pool_xy[1] = 0
                if pool_xy[1] > height - 1:
                    # print(xy[1], pool_xy[1], height - 1)
                    pool_xy[1] = height - 1

                sp_info[kk] = np.copy(padded[pool_xy[1]: pool_xy[1]+(2*SP['patch_r']+1),\
                                             pool_xy[0]: pool_xy[0]+(2*SP['patch_r']+1)].ravel())

                assert(len(sp_info[kk])==feature_len)

                bcs = SP['b_cover_size']
                cover_msk[max(0,int(pool_xy[1]-bcs//2)):min(height,int(pool_xy[1]+bcs//2+1)),\
                          max(0,int(pool_xy[0]-bcs//2)):min(width,int(pool_xy[0]+bcs//2+1))] = 1

            instance_info[mm] = sp_info

        bg_r,bg_c = np.where(cover_msk==0)
        if len(bg_r) > SP['back_per_instance']*10:
            anno_file2 = os.path.join(Dataset['anno_dir'].format(category), '{}.mat'.format(img_list[nn]))
            matcontent2 = sio.loadmat(anno_file2)
            bbox_value = matcontent2['record']['objects'][0,0][0,int(idx_list[nn])-1]['bbox'][0]*resize_ratio
            # bbox_value_pool = (bbox_value//Astride).astype(int)
            bbox_value_pool = np.round((bbox_value - 8.5) / 16).astype(np.int32) 
            cover_msk_inner = np.ones_like(cover_msk)
            cover_msk_inner[bbox_value_pool[1]:bbox_value_pool[3]+1, bbox_value_pool[0]:bbox_value_pool[2]+1] = 0
            bg_ri,bg_ci = np.where(np.logical_or(cover_msk,cover_msk_inner)==0)
            
            bg_ro,bg_co = np.where(cover_msk_inner==1)
            
            if len(bg_ri) > SP['back_per_instance']*5:
                bg_select1 = np.random.choice(len(bg_ri), size=(SP['back_per_instance'],), replace=False)
            else:
                bg_select1 = np.array([])
                
            if len(bg_ro) > SP['back_per_instance']*10:
                bg_select2 = np.random.choice(len(bg_ro), size=(SP['back_per_instance'],), replace=False)
            else:
                bg_select2 = np.array([])
            
        else:
            bg_select1 = np.array([])
            bg_select2 = np.array([])

        bg_infoi = [None for bb in range(len(bg_select1))]
        for bgi, bg_idx in enumerate(bg_select1):
            bg_rr = bg_ri[bg_idx]
            bg_cc = bg_ci[bg_idx]

            bg_infoi[bgi] = np.copy(padded[bg_rr: bg_rr+(2*SP['patch_r']+1),bg_cc:bg_cc+(2*SP['patch_r']+1)].ravel())
            
        instance_info[sp_num] = bg_infoi
            
        bg_infoo = [None for bb in range(len(bg_select2))]
        for bgi, bg_idx in enumerate(bg_select2):
            bg_rr = bg_ro[bg_idx]
            bg_cc = bg_co[bg_idx]

            bg_infoo[bgi] = np.copy(padded[bg_rr: bg_rr+(2*SP['patch_r']+1),bg_cc:bg_cc+(2*SP['patch_r']+1)].ravel())

        instance_info[sp_num+1] = bg_infoo

        all_info[nn] = instance_info


    save_name = os.path.join(VCencode_dir, 'SP_{}_encoding_{}.pickle'.format(VC['layer'], category))
    with open(save_name,'wb') as fh:
        pickle.dump(all_info, fh)
        
    
if __name__=='__main__':
    for category in all_categories2[3:4]:
        get_SP_VCencoding(category)
