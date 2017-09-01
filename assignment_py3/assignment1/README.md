Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.

作业总体流程是，以ipynb文件为主，补充好classifiers文件夹的代码，使得ipynb运行流畅
参考[代码讲解](https://zhuanlan.zhihu.com/c_115985719) [问答题](http://minghaowu.tech/category/cs231n/)
# CS231n : Deep Learning Lecture for Computer Vision
Stanford CS231n 2016 Assignment

## Assignment1

### Q1: k-Nearest Neighbor classifier (Completed)
- X和X_train比较，最好画出数据框图，能清晰地反映计算的流程。
- 文件提到3种计算欧式距离的方法，将公式分解后使用矩阵乘法是最快的。
- 应当熟悉numpy的ndarray对象，它与list、tuple不一样，代码有许多处理张量维度的函数，多维数组转为一维数组的办法是添加[:,0]
### Q2: Training a Support Vector Machine (Completed)
- 画矩阵相乘的数据框图，末尾加上折叶损失函数
- 首次接触反向传播以及梯度下降，折叶损失相当于路由器，根据损失函数细心推导
- 向量化的技巧，是否需要传播梯度可以用掩码矩阵，不用使用if语句
- 数值梯度检验，利用到lamda匿名函数
- 交叉验证时，设立两个值min和max，使用arange(min,max,(max-min)/time)测试多个超参数，不用手写数组元素
- 权重可视化，因为第一层权重相当于图像大小，将权重值映射到0-255区间，可以观察到模板信息
### Q3: Implement a Softmax classifier (Completed)
- 画矩阵相乘框图，末尾加上交叉熵损失函数
- 求导方式见[博客](http://blog.csdn.net/u014313009/article/details/51045303)
- python的代码格式有错，导致多加一个循环，应当注意
### Q4: Two-Layer Neural Network (Completed)
- python不需要声明类型，直接给变量赋值即可，但有些地方提前给变量赋值，方便阅读代码

### Q5: Higher Level Representations: Image Features (Completed)
- 提取图片特征再将特征当作svm或者two layer networks的输入图片，进行预测

## Assignment2

### Q1: Fully-connected Neural Network (Completed)

### Q2: Batch Normalization (Completed)

### Q3: Dropout (Completed)

### Q4: ConvNet on CIFAR-10 (A Little Bit Left)



## Assignment3

### Q1: Image Captioning with Vanilla RNNs (Completed)

### Q2: Image Captioning with LSTMs (Completed)

### Q3: Image Gradients: Saliency maps and Fooling Images (Not Yet)

### Q4: Image Generation: Classes, Inversion, DeepDream (Not Yet)


