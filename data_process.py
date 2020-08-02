'''
cv2.resize(img,shape=())
nterpolation - 插值方法。共有5种：
１）INTER_NEAREST - 最近邻插值法
２）INTER_LINEAR - 双线性插值法（默认）
３）INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
４）INTER_CUBIC - 基于4x4像素邻域的3次插值法
５）INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值

img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)

'''

import numpy as np
import os
import cv2


data_path = './dataset/positive/'
data_save = './dataset/'
def process():
    imgs = []
    imgs_gray = []
    files = os.listdir(data_path)
    print('load data ...')
    for ei, file in enumerate(files):
        I = cv2.imread(data_path + file)
        I = cv2.resize(I, dsize=(800, 600), interpolation=cv2.INTER_LINEAR)
        I_gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        imgs.append(I)
        imgs_gray.append(I_gray)
        # print(I.shape)
    np.save(data_save + 'data_color.npy', imgs)
    np.save(data_save + 'data_gray.npy', imgs_gray)

def load_data(dtype='gray', reloading=False):
    if not os.path.exists(data_save + 'data_' + dtype + '.npy') or reloading == True:
        process()
    return np.load(data_save + 'data_' + dtype + '.npy')

# process()
