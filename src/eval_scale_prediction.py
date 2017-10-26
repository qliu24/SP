from config import *
import scipy.io as sio

def eval_scale_prediction(category, set_type):
    img_dir = Dataset['img_dir_org'].format(category)
    anno_dir = Dataset['anno_dir'].format(category)
    filelist = Dataset['{}_list'.format(set_type)].format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    img_list = [cc.strip().split()[0] for cc in contents if cc != '\n']
    idx_list = [cc.strip().split()[1] for cc in contents if cc != '\n']
    N = len(img_list)
    print('Total image number for {} set of {}: {}'.format(set_type, category, N))
    
    scale_save_file = os.path.join(Score_dir, 'scale_{}_{}.pickle'.format(category, set_type))
    with open(scale_save_file,'rb') as fh:
        scale_record = pickle.load(fh)
        
    loss_ls = []
    for nn in range(N):
        if nn%100==0:
            print(nn)
            
        img_file = os.path.join(img_dir, '{}.JPEG'.format(img_list[nn]))
        img = cv2.imread(img_file)
        
        img_h, img_w = img.shape[0:2]
        anno_file = os.path.join(anno_dir, '{}.mat'.format(img_list[nn]))
        matcontent = sio.loadmat(anno_file)
        
        bbox_value = matcontent['record']['objects'][0,0][0,int(idx_list[nn])-1]['bbox'][0]
        bbox_value = [max(math.ceil(bbox_value[0]), 1), max(math.ceil(bbox_value[1]), 1), \
                        min(math.floor(bbox_value[2]), img_w), min(math.floor(bbox_value[3]), img_h)]
        
        bbox_height = bbox_value[3]-bbox_value[1]+1
        bbox_width = bbox_value[2]-bbox_value[0]+1
        resize_ratio = scale_size/np.min((bbox_height,bbox_width))
        
        scale_gt = resize_ratio*np.min((img_h, img_w))
        # print(scale_gt)
        loss_ls.append(np.log(np.max((scale_record[nn], scale_gt))/np.min((scale_record[nn], scale_gt))))
        
    print('scale loss: {}'.format(np.mean(loss_ls)))
    
if __name__=='__main__':
    for category in all_categories2[3:4]:
        eval_scale_prediction(category, set_type='test')