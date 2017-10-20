from config import *

for category in all_categories2:
    fname=Dict['Dictionary'].format(category).replace('.pickle','_example.pickle')
    with open(fname,'rb') as fh:
        example = pickle.load(fh)

    print(len(example))
    save_dir = os.path.join(root_dir, 'dictionary','example_{}_K{}_{}'.format(VC['layer'], VC['num'], category))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for ii in range(len(example)):
        big_img = np.zeros((10+(Arf+10)*4, 10+(Arf+10)*5, 3))
        for iis in range(20):
            if iis >= example[ii].shape[1]:
                continue

            aa = iis//5
            bb = iis%5
            rnum = 10+aa*(Arf+10)
            cnum = 10+bb*(Arf+10)
            big_img[rnum:rnum+Arf, cnum:cnum+Arf, :] = example[ii][:,iis].reshape(Arf,Arf,3).astype('uint8')

        save_file = os.path.join(save_dir, 'example_K{}.png'.format(ii))
        cv2.imwrite(save_file, big_img)

