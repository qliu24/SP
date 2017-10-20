from config import *
import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import datetime
from copy import *
from FeatureExtractor import FeatureExtractor

# category = 'car'
extractor = FeatureExtractor(cache_folder=model_cache_folder, which_net='vgg16', which_layer=VC['layer'], which_snapshot=0)
for category in all_categories2:
    check_num = 1000  # save how many images to one file
    samp_size = 50  # number of features per image

    # Specify the dataset
    img_dir = Dataset['img_dir'].format(category)
    filelist = Dataset['train_list'].format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    image_path = [os.path.join(img_dir, '{}_{}.JPEG'.format(cc.strip().split()[0], cc.strip().split()[1])) for cc in contents if cc != '\n']
    if category != 'car':
        image_path += [mm.replace('.JPEG','_p.JPEG') for mm in image_path]

    img_num = len(image_path)
    print('total number of images for {}: {}'.format(category, img_num))

    res = np.zeros((featDim, 0))
    loc_set = np.zeros((5, 0))

    for ii in range(img_num):
        assert(os.path.exists(image_path[ii]))
        img = cv2.imread(image_path[ii])
        # img = cv2.resize(img, (scale_size, scale_size))
        # img = myresize(img, scale_size, 'short')
        assert(np.min(img.shape[0:2])==224)

        tmp = extractor.extract_feature_image(img)[0]
        assert(tmp.shape[2]==featDim)
        height, width = tmp.shape[0:2]
        tmp = tmp[offset:height - offset, offset:width - offset, :]
        ntmp = np.transpose(tmp, (2, 0, 1))
        gtmp = ntmp.reshape(ntmp.shape[0], -1)
        if gtmp.shape[1] >= samp_size:
            rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size]
        else:
            rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size-gtmp.shape[1]]
            rand_idx = np.append(range(gtmp.shape[1]), rand_idx)

        res = np.column_stack((res, deepcopy(gtmp[:, rand_idx])))
        for rr in rand_idx:
            ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
            hi = Astride * (ihi + offset) - Apad
            wi = Astride * (iwi + offset) - Apad
            assert (hi >= 0)
            assert (hi <= img.shape[0] - Arf)
            assert (wi >= 0)
            assert (wi <= img.shape[1] - Arf)
            loc_set = np.column_stack((loc_set, [ii, hi, wi, hi+Arf, wi+Arf]))

        if (ii + 1) % check_num == 0 or ii == img_num - 1:
            print('saving batch {0}/{1}'.format(ii//check_num+1, math.ceil(img_num/check_num)))
            # fnm = Dict['cache_path_sub']+'{}_set{}.pickle'.format(ii//check_num, subset_idx)
            fnm = Dict['cache_path'].format(category)+'{}.pickle'.format(ii//check_num)
            with open(fnm, 'wb') as fh:
                pickle.dump([res, loc_set, image_path], fh)

            res = np.zeros((featDim, 0))
            loc_set = np.zeros((5, 0))

        if ii%50==0:
            print(ii, end=' ', flush=True)

