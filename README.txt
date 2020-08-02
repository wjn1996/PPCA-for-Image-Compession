# finished by 王嘉宁
# email:lygwjn@126.com
# time:2020.06.26
#
源程序使用说明：
（1）目录：
dataset存放NWPU VHR-10的原始数据集文件，需自行下载（http://www.escience.cn/people/gongcheng/NWPU-VHR-10.html）；
save存放压缩后的数据，需事先定义该目录
case存放某一张图像在不同算法下压缩后恢复的图像，需事先定义该目录
curve存放实验结果曲线图，需事先定义该目录

main.py 程序入口文件
compression.py 算法类文件
data_process.py 数据预处理和加载文件
plot.py 实验结果可视化文件

（2）算法执行：
执行命令：控制台输入：
python main.py <method> <color>

其中<method>共有四种模型，分别是：
'imageset_compression':传统的PCA算法，对图像数据集数量压缩
'pixel_compression':传统的PCA算法，对图像数据集像素个数进行压缩
'Two_D_compression':2DPCA算法
'bounds_compression':P-PCA算法（分片PCA，本文提出的方法）
<color>表示图像类型，取值为gray时为灰度图，color时为彩色图。

例如执行：
python main.py bounds_compression color
即为使用P-PCA算法对彩色图像进行压缩，可以在main函数中修改分片数bounds_num，默认为30。

（3）实验结果
执行如下命令
python plot.py
即可生成两个实验的结果图。需要注意的是，比如对四个算法分别在灰度图和彩色图执行完毕后才能执行，否则因为没有生成必要的结果文件而报错。