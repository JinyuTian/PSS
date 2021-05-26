# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # Jpeg augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name,c):
    # get multiscale patches from a single image
    img = cv2.imread(file_name)  # gray scale
    index = np.random.randint(0,np.min([img.shape[0],img.shape[1]])-180)
    img = img[index:index+180,index:index+180]
    if c<3:
        img = img[:,:,c]
        h, w = img.shape
    else:
        img = img
        h, w,_ = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    if x_aug == []:
                        print()
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='Jpeg/Train400', verbose=False,channel=0):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.JPEG')[0:20] # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        try:
            patches = gen_patches(file_list[i],channel)
        except:
            continue
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    # print('^_^-training Jpeg finished-^_^')
    return data


def Val_gen_patches(x,stride=40):
    # get multiscale patches from a single image
    # x2 = transforms.ToTensor()(im1)
    if x.shape[0]<x.shape[1]:
        diff = int(abs(x.shape[1] - x.shape[0]))
        Pad_size = int(np.floor(diff/2))
        img = np.pad(x,((Pad_size,diff-Pad_size),(0,0),(0,0)),'symmetric')
    else:
        diff = int(abs(x.shape[1] - x.shape[0]))
        Pad_size = int(np.floor(diff/2))
        img = np.pad(x, ((0, 0), (Pad_size,diff-Pad_size), (0, 0)), 'symmetric')
    if not img.shape[0]==img.shape[1]:
        print('wori')

    Pstride = 40
    if not round(img.shape[0]/Pstride) == img.shape[0]/Pstride:
        diff = int(np.ceil(img.shape[0]/Pstride)*Pstride - img.shape[0])
        Pad_size = int(np.floor(diff / 2))
        img = np.pad(img, ((0, 0), (Pad_size,diff-Pad_size), (0, 0)), 'symmetric')
        img = np.pad(img,((Pad_size,diff-Pad_size),(0,0),(0,0)),'symmetric')

    h,w,c = img.shape
    patches = []
    from PIL import Image

    img = Image.fromarray(np.uint8(img))
    img_scaled = np.asarray(img)
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img_scaled[i:i+patch_size, j:j+patch_size]
            for k in range(0, aug_times):
                x_aug = data_aug(x, mode=0)
                patches.append(x_aug)
    return patches,img_scaled

def test_datagenerator(ori_data_dir='', channel=0,verbose=False,Rate_Pathes=0.5,seed=10,file_list='',stride=40,train=1):
    # generate clean patches from a dataset
    ori_file_list = glob.glob(ori_data_dir+'/*.png') # get name list of all .png files
    ori_file_list = sorted(ori_file_list)
    # initrialize
    dataXs = []
    ori_shape_list = []
    ori_x = []
    ori_extend_x = []
    # generate patches
    for i,ori_img in enumerate(ori_file_list):
        import cv2
        x = cv2.imread(ori_img)  # gray scale
        ori_shape = x.shape
        ori_x.append(x)
        xs,extend_x = Val_gen_patches(x,stride=stride)
        ori_extend_x.append(extend_x)
        xs = np.array(xs, dtype='uint8')
        dataXs.append(xs)
        ori_shape_list.append(ori_shape)

    # print('^_^-testing Jpeg finished-^_^')
    return dataXs,ori_shape_list,ori_x,ori_extend_x,ori_file_list


if __name__ == '__main__':

    data = datagenerator(data_dir='Jpeg/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving Jpeg...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')