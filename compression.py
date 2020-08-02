'''
n_components:[0-1],int, 'mle'
'''


import numpy as np
import os
import cv2
from data_process import load_data
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
# 注：默认输入的图像x都是三个维度[N H W]，如果是彩色三通道图像，则首先展开成[N H 3*W]。

class PCA_compression():
    def __init__(self, x, bounds_num):
        self.x = x
        self.sample_num = self.x.shape[0]
        self.height = self.x.shape[1]
        self.width = x.shape[2]
        self.bounds_num = bounds_num
        self.path = 'E:/华师校内课程及实验/2020数据科学与工程算法基础/project/'

    def setVarianceRate(self, variance_rate=0.95):
        self.variance_rate = variance_rate

    def save(self, newx):
        # 将压缩的图像存储起来
        if not os.path.exists(self.path + 'save'):
            os.makedirs(self.path + 'save')
        np.save(self.path + 'save/' + self.name + '.npy', newx)

    def MSE(self, X, X_hat):
        # MSE
        return np.mean(np.square(X - X_hat))

    def PSNR(self, X, X_hat):
        return 10* np.log(255*2 / (np.mean(np.square(X - X_hat))))
 
 
    def SSIM(self, X, X_hat):
        u_true = np.mean(X_hat)
        u_pred = np.mean(X)
        var_true = np.var(X_hat)
        var_pred = np.var(X)
        std_true = np.sqrt(var_true)
        std_pred = np.sqrt(var_pred)
        c1 = np.square(0.01*7)
        c2 = np.square(0.03*7)
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        return ssim / denom



    def imageset_compression(self):
        self.name = 'imageset_compression'
        # 根据一个图像集合，根据图像之间的相关性压缩图像的数量，但像素个数不变
        # x:[N,W,H]
       
        x = self.x.reshape(self.sample_num, -1) # x:[N,W*H] 行为样本数，列为像素数，图像压缩目标是压缩像素
        x = x.T # x:[W*H,N] 列为样本数，行为像素数，图像压缩目标是压缩样本
        print('x.shape=', x.shape)
        pca = PCA(n_components=self.variance_rate)
        newx = pca.fit_transform(x)
        self.save(newx)
        ratio = pca.explained_variance_ratio_
        n_components_ = pca.n_components_
        com_ratio = float((n_components_*self.width*self.height + (n_components_ + 1)*self.sample_num)/(self.sample_num*self.width*self.height))
        # print('pca.explained_variance_ratio_=', ratio)
        print('newx.shape=', newx.shape)
        print('n_components_=', n_components_)
        print('compression ratio=', com_ratio)
        # print('one-image compression ratio=', com_ratio)
        # 恢复
        X = pca.inverse_transform(newx)
        print('X.shape=', X.shape)
        X = X.T
        X = X.reshape(self.sample_num, self.height, self.width)
        return X, n_components_, com_ratio



    def pixel_compression(self):
        self.name = 'pixel_compression'
        # 根据一个图像的集合，考虑到不同图像之间以及不同像素点之间的相关性，压缩像素的个数，但不改变图像的数量
        # x:[N,W,H]
       
        x = self.x.reshape(self.sample_num, -1) # x:[N,W*H] 行为样本数，列为像素数，图像压缩目标是压缩像素
        print('x.shape=', x.shape)
        pca = PCA(n_components=self.variance_rate)
        newx = pca.fit_transform(x)
        self.save(newx)
        ratio = pca.explained_variance_ratio_
        n_components_ = pca.n_components_
        com_ratio = float((newx.shape[0]*newx.shape[1] + (n_components_ + 1)*self.sample_num)/(x.shape[0]*x.shape[1]))
        # print('pca.explained_variance_ratio_=', ratio)
        print('newx.shape=', newx.shape)
        print('n_components_=', n_components_)
        print('compression ratio=', com_ratio)
        # print('one-image compression ratio=', com_ratio)
        # 恢复
        X = pca.inverse_transform(newx)
        print('X.shape=', X.shape)
        X = X.reshape(self.sample_num, self.height, self.width)
        return X, n_components_, com_ratio

    def Two_D_compression(self):
        self.name = 'Two_D_compression'
        #2DPCA降维，直接对每个图像降低列数
        #imgs 是三维的图像矩阵，第一维是图像的个数

        a, b, c = self.sample_num, self.height, self.width
        average = np.zeros((b,c))
        for i in range(a):
            average += self.x[i,:,:]/(a*1.0)
        G_t = np.zeros((c,c))
        for j in range(a):
            img = self.x[j,:,:]
            temp = img-average
            G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
        w,v = np.linalg.eigh(G_t)
        w = w[::-1]
        v = v[::-1]
        for k in range(c):
            alpha = sum(w[:k])*1.0/sum(w)
            if alpha >= self.variance_rate:
                u = v[:,:k] # 特征向量
                break
        u = np.array(u)
        newx = np.dot(self.x-average, u)# 压缩后的图像
        self.save(newx)
        # print('u.shape=', np.array(u).shape) 800*67
        ratio = float(a*u.shape[1]*b + u.shape[0]*u.shape[1])/(a*b*c)
        print('n_components_=', u.shape[1])
        print('compression ratio=', ratio)
        #还原
        X = np.dot(newx, u.T) + average
        return X, u.shape[1], ratio

    def bounds_compression(self):
        self.name = 'bounds_compression'
        # 不同于前两个，本方法可以为每张图像单独压缩，为每个图像按行/列划分每一个bounds，每个bounds内每一行进行压缩。
        # x:[N,H,W]
        row_num = self.height//self.bounds_num
        num = 0
        memory = 0
        n_com = 0
        new_imgs = np.array([])

        # 遍历每张图像
        for i in range(self.sample_num):
            x = self.x[i].reshape(self.height, -1)
            new_img = np.array([])
            # 遍历每一个bound
            for j in range(self.bounds_num):
                num += 1
                b = x[j*row_num:(j+1)*row_num]
                pca = PCA(n_components=self.variance_rate)
                # print(j)
                newx = pca.fit_transform(b)
                memory += newx.shape[0]*newx.shape[1] + (pca.n_components_ + 1)*row_num
                n_com += pca.n_components_
                X = pca.inverse_transform(newx)
                if j==0:
                    new_img = X
                else:
                    new_img = np.append(new_img, X, 0)
            if i==0:
                new_imgs = new_img
            else:
                new_imgs = np.append(new_imgs, new_img, 0)
        ratio = float(memory)/(self.height*self.width*self.sample_num)
        print('n_components_ (avg)=', n_com/num)
        print('compression ratio (avg)=', ratio)
        return new_imgs.reshape(self.sample_num, self.height, self.width), int(n_com/num), ratio




