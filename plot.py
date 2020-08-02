import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 记录训练和测试过程中的loss和acc变化
save_path = './save/'
imageset_compression_color = np.load(save_path + 'imageset_compression_color.npz')
pixel_compression_color = np.load(save_path + 'pixel_compression_color.npz')
Two_D_compression_color = np.load(save_path + 'Two_D_compression_color.npz')
bounds_compression_color = np.load(save_path + 'bounds_compression_color.npz')

imageset_compression_gray = np.load(save_path + 'imageset_compression_gray.npz')
pixel_compression_gray = np.load(save_path + 'pixel_compression_gray.npz')
Two_D_compression_gray = np.load(save_path + 'Two_D_compression_gray.npz')
bounds_compression_gray = np.load(save_path + 'bounds_compression_gray.npz')

#### 实验一（1）：整体数据集上不同算法的压缩率比较


#### 实验一（2）：某一张图像上，不同算法在不同的阈值下MSE、PSNR和SSIM比较

###MSE###
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)
ax1.set(xlim=[0.5,1], ylim=[0,2000], title='不同算法的MSE比较（灰度图）', ylabel='MSE', xlabel='variance_rate')
ax2.set(xlim=[0.5,1], ylim=[0,3500], title='不同算法的MSE比较（彩色图）', ylabel='MSE', xlabel='variance_rate')
ax1.plot([i for i in imageset_compression_gray['variance_rate']], imageset_compression_gray['mse'])
ax1.plot([i for i in pixel_compression_gray['variance_rate']], pixel_compression_gray['mse'])
ax1.plot([i for i in Two_D_compression_gray['variance_rate']], Two_D_compression_gray['mse'])
ax1.plot([i for i in bounds_compression_gray['variance_rate']], bounds_compression_gray['mse'])
ax2.plot([i for i in imageset_compression_gray['variance_rate']], imageset_compression_color['mse'])
ax2.plot([i for i in pixel_compression_gray['variance_rate']], pixel_compression_color['mse'])
ax2.plot([i for i in Two_D_compression_gray['variance_rate']], Two_D_compression_color['mse'])
ax2.plot([i for i in bounds_compression_gray['variance_rate']], bounds_compression_color['mse'])
ax1.legend(('PCA', 'Pixel-PCA', '2DPCA', 'P-PCA'))
plt.savefig("./curve/mse.png")

###PSNR###
fig2 = plt.figure()
ax1 = fig2.add_subplot(121)
ax2 = fig2.add_subplot(122)
ax1.set(xlim=[0.5,1], ylim=[-20,50], title='不同算法的PSNR比较（灰度图）', ylabel='PSNR', xlabel='variance_rate')
ax2.set(xlim=[0.5,1], ylim=[-20,50], title='不同算法的PSNR比较（彩色图）', ylabel='PSNR', xlabel='variance_rate')
ax1.plot([i for i in imageset_compression_gray['variance_rate']], imageset_compression_gray['psnr'])
ax1.plot([i for i in pixel_compression_gray['variance_rate']], pixel_compression_gray['psnr'])
ax1.plot([i for i in Two_D_compression_gray['variance_rate']], Two_D_compression_gray['psnr'])
ax1.plot([i for i in bounds_compression_gray['variance_rate']], bounds_compression_gray['psnr'])
ax2.plot([i for i in imageset_compression_gray['variance_rate']], imageset_compression_color['psnr'])
ax2.plot([i for i in pixel_compression_gray['variance_rate']], pixel_compression_color['psnr'])
ax2.plot([i for i in Two_D_compression_gray['variance_rate']], Two_D_compression_color['psnr'])
ax2.plot([i for i in bounds_compression_gray['variance_rate']], bounds_compression_color['psnr'])
ax1.legend(('PCA', 'Pixel-PCA', '2DPCA', 'P-PCA'))
plt.savefig("./curve/psnr.png")
###SSIM###
fig3 = plt.figure()
ax1 = fig3.add_subplot(121)
ax2 = fig3.add_subplot(122)
ax1.set(xlim=[0.5,1], ylim=[0,1.05], title='不同算法的SSIM比较（灰度图）', ylabel='SSIM', xlabel='variance_rate')
ax2.set(xlim=[0.5,1], ylim=[0.25,1.05], title='不同算法的SSIM比较（彩色图）', ylabel='SSIM', xlabel='variance_rate')
ax1.plot([i for i in imageset_compression_gray['variance_rate']], imageset_compression_gray['ssim'])
ax1.plot([i for i in pixel_compression_gray['variance_rate']], pixel_compression_gray['ssim'])
ax1.plot([i for i in Two_D_compression_gray['variance_rate']], Two_D_compression_gray['ssim'])
ax1.plot([i for i in bounds_compression_gray['variance_rate']], bounds_compression_gray['ssim'])
ax2.plot([i for i in imageset_compression_gray['variance_rate']], imageset_compression_color['ssim'])
ax2.plot([i for i in pixel_compression_gray['variance_rate']], pixel_compression_color['ssim'])
ax2.plot([i for i in Two_D_compression_gray['variance_rate']], Two_D_compression_color['ssim'])
ax2.plot([i for i in bounds_compression_gray['variance_rate']], bounds_compression_color['ssim'])
ax1.legend(('PCA', 'Pixel-PCA', '2DPCA', 'P-PCA'))
plt.savefig("./curve/ssim.png")

# print(bounds_compression_gray['psnr'])

#### 实验二（2）：某一张图像上，P-PCA算法在不同压缩率下PSNR、SSIM值比较
fig4 = plt.figure()
ax1 = fig4.add_subplot(121)
ax2 = fig4.add_subplot(122)
ax1.set(xlim=[0.4,3], ylim=[10,50], title='P-PCA算法PSNR变化', ylabel='PSNR', xlabel='compression_rate(%)')
ax1.plot([i*100 for i in bounds_compression_gray['cr']], bounds_compression_gray['psnr'])
ax1.plot([i*100 for i in bounds_compression_gray['cr']], bounds_compression_color['psnr'])
ax1.legend(('灰度图', '彩色图'))
ax2.set(xlim=[0.4,3], ylim=[0.998,1], title='P-PCA算法SSIM变化', ylabel='SSIM', xlabel='compression_rate(%)')
ax2.plot([i*100 for i in bounds_compression_gray['cr']], bounds_compression_gray['ssim'])
ax2.plot([i*100 for i in bounds_compression_gray['cr']], bounds_compression_color['ssim'])
ax2.legend(('灰度图', '彩色图'))
plt.savefig("./curve/p-pca.png")


# print(train_loss, '\n', train_acc, '\n', test_loss, '\n', test_acc)
# train_loss_ = [train_loss[i] for i in range(0, len(train_loss), 100)]
# train_acc_ = [train_acc[i] for i in range(0, len(train_acc), 100)]


# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax1.set(xlim=[0,len(people_SNN)], ylim=[0, 1], title='face recogintion acc',ylabel='acc(%)', xlabel='test times')
# ax1.plot([i for i in range(len(people_SNN))], people_SNN)
# ax1.plot([i for i in range(len(people_SNN2))], people_SNN2)
# ax1.plot([i for i in range(len(people_CNN1))], people_CNN1)
# ax1.plot([i for i in range(len(people_CNN2))], people_CNN2)
# ax1.legend(('single-layer SNN', 'multi-layer-SNN', '2-layer CNN', '4-layer CNN'))

# ax2.set(xlim=[0,len(mood_SNN)], ylim=[0, 1], title='mood recogintion acc',ylabel='acc(%)', xlabel='test times')
# ax2.plot([i for i in range(len(mood_SNN))], mood_SNN)
# ax2.plot([i for i in range(len(mood_SNN2))], mood_SNN2)
# ax2.plot([i for i in range(len(mood_CNN1))], mood_CNN1)
# ax2.plot([i for i in range(len(mood_CNN2))], mood_CNN2)
# ax2.legend(('single-layer SNN', 'multi-layer-SNN', '2-layer CNN', '4-layer CNN'))
plt.show()
