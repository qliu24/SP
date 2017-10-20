import numpy as np
import cv2
import scipy.io as sio
import os
import matplotlib.pyplot as plt

def fix_img(category, ff):
    image_f = '/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1/Images/{}_imagenet/{}.JPEG'.format(category,ff)
    img=cv2.imread(image_f)
    plt.imshow(img[:,:,::-1])
    plt.show()
    img_fix = np.transpose(img, (1,0,2))
    img_fix2 = cv2.flip(img_fix, 0)
    plt.imshow(img_fix2[:,:,::-1])
    plt.show()
    return (image_f,img_fix2)


all_categories = ['aeroplane','bicycle', 'bus', 'car', 'motorbike', 'train']
for category in all_categories:
    print(category)
    for set_type in ['train','test']:
        file_list_f = '/mnt/1TB_SSD/dataset/PASCAL3D+_sp/file_list/{0}_list/{1}_{0}.txt'.format(set_type,category)
        with open(file_list_f,'r') as fh:
            content = fh.readlines()

        file_list = [cc.strip().split()[0] for cc in content if cc!='\n']
        print(len(file_list))
        for ff in file_list:
            image_f = '/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1/Images/{}_imagenet/{}.JPEG'.format(category,ff)
            img=cv2.imread(image_f)

            anno_file = '/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat'.format(category,ff)
            matcontent2 = sio.loadmat(anno_file)
            ww = matcontent2['record']['size'][0,0]['width'][0,0][0,0]
            hh = matcontent2['record']['size'][0,0]['height'][0,0][0,0]

            try:
                assert(ww == img.shape[1])
                assert(hh == img.shape[0])
            except:
                print(category, ff, hh, ww, img.shape)

        print('done')