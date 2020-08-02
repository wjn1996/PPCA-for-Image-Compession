import sys
from data_process import load_data
from compression import *
import matplotlib.pyplot as plt
import datetime

bounds_num = 30 # 一张图像一个块的行数
alg = sys.argv[1]
color = sys.argv[2]

if color not in ['color', 'gray']: # gray color
    color = 'gray'
data = load_data(dtype=color, reloading=True)
ax = plt.subplot(1, 2, 1)
ax.imshow(data[1])
ax.set_title('orgin')

data = data.reshape(data.shape[0], data.shape[1], -1)
print('data num=', data.shape[0])
print('image shape=(', data.shape[1], ',', data.shape[2], ')')
pca = PCA_compression(data[:50], bounds_num)

# 所有算法列表
algorithm_list = {
    'imageset_compression':pca.imageset_compression,
    'pixel_compression':pca.pixel_compression,
    'bounds_compression':pca.bounds_compression,
    'Two_D_compression':pca.Two_D_compression
}
if alg not in algorithm_list.keys():
    print('error')
    exit()


algorithm = algorithm_list[alg]
print('algorithm name:', alg)

variance_rate = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
k = [] #成分个数
cr = [] # 压缩率
mse = [] # 均方误差
psnr = [] # psnr
ssim = [] # ssim
for i in variance_rate:
    print('========variance_rate(=' + str(i) + ')===========')
    pca.setVarianceRate(i)
    starttime = datetime.datetime.now()
    X, n_components_, ratio = algorithm()
    
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    print('runnig time:', duringtime.seconds, '(s)')
    MSE_loss = pca.MSE(data[1], X[1])
    PSNR_loss = pca.PSNR(data[1], X[1])
    SSIM_loss = pca.SSIM(data[1], X[1])
    
    print('MSE loss=', MSE_loss, '\tPSNR loss=', PSNR_loss, '\nSSIM_loss=', SSIM_loss)

    if color == 'color':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]//3, 3).astype(np.uint8)

    ax = plt.subplot(1, 2, 2)
    ax.imshow(X[1])
    ax.set_title('compression')
    plt.savefig('./case/case_' + alg + '(vr=' + str(i) + ',color=' + color + ').jpg')

    k.append(n_components_)
    cr.append(ratio)
    mse.append(MSE_loss)
    psnr.append(PSNR_loss)
    ssim.append(SSIM_loss)
np.savez("./save/" + alg + "_" + color + ".npz", variance_rate=variance_rate, k=k, cr=cr, mse=mse, psnr=psnr, ssim=ssim)
