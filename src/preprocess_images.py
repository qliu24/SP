from config import *
import scipy.io as sio

set_type = 'test'

for category in all_categories2:
    img_dir = Dataset['img_dir_org'].format(category)
    anno_dir = Dataset['anno_dir'].format(category)
    filelist = Dataset['{}_list'.format(set_type)].format(category)
    save_dir = Dataset['img_dir'].format(category)

    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    file_list = [cc.strip().split()[0] for cc in contents if cc != '\n']
    idx_list = [int(cc.strip().split()[1]) for cc in contents if cc != '\n']
    img_num = len(file_list)
    print('total number of images for {}: {}'.format(category, img_num))

    for ii in range(img_num):
        if not file_list[ii]:
            print(ii)
            continue

        save_name = os.path.join(save_dir, '{}_{}.JPEG'.format(file_list[ii], idx_list[ii]))
        assert(not os.path.exists(save_name))

        img_file = os.path.join(img_dir, '{}.JPEG'.format(file_list[ii]))
        try:
            assert(os.path.exists(img_file))
        except:
            print('file not exist: {}'.format(img_file))
            continue

        img = cv2.imread(img_file)
        img_h, img_w = img.shape[0:2]

        anno_file = os.path.join(anno_dir, '{}.mat'.format(file_list[ii]))
        try:
            assert(os.path.exists(anno_file))
        except:
            print('file not exist: {}'.format(anno_file))
            continue

        matcontent = sio.loadmat(anno_file)
        bbox = matcontent['record']['objects'][0,0][0,idx_list[ii]-1]['bbox'][0]
        # print(bbox)
        bbox = np.array([max(1,np.floor(bbox[0])), max(1,np.floor(bbox[1])), \
                         min(img_w, np.ceil(bbox[2])), min(img_h, np.ceil(bbox[3]))]).astype(int)
        patch = img[bbox[1]-1:bbox[3], bbox[0]-1:bbox[2], :]

        patch = myresize(patch, scale_size, 'short')
        cv2.imwrite(save_name, patch)

        # augment data
        if set_type=='train' and category != 'car':
            patch = cv2.flip(patch, 1)
            cv2.imwrite(save_name.replace('.JPEG','_p.JPEG'), patch)

    