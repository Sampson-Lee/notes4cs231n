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
- .py扩展名的文件是源代码文件，.pyc扩展名的是python的编译文件，.pyd文件并不是使用python编写而成，.pyd文件一般是其他语言编写的python扩展模块。
- 注意ipython的kernel是3.5，编译外部文件都应该跳转到py35环境，im2col是第二次掉环境坑！
### Q5: Higher Level Representations: Image Features (Completed)
- 提取图片特征再将特征当作svm或者two layer networks的输入图片，进行预测

## Assignment2

### Q1: Fully-connected Neural Network (Completed)
- relu层的前向传播代码为`out=x*(x>=0)`
- solver解释[CS231n Solver.py 详解](http://www.cnblogs.com/lijiajun/p/5582789.html)
- 多次因写错代码而使得优化过程出错
### Q2: Batch Normalization (Completed)
- 理解BN的作用是增大网络的容量，[神经网络的容量](https://pure-earth-7284.herokuapp.com/2016/09/07/神经网络的容量/)
- 在读取dict的key和value时，如果key不存在，就会触发KeyError错误
- 漏写self.params导致keyerror，这些脑残问题可以直接在google出来的！
- 仍旧是keyerror问题，是因为混淆循环语句而导致关键字出错，不能只靠检查代码，而要print数据
- 写代码时等号左边不能随意换行，等号右边可以
### Q3: Dropout (Completed)
- 在训练过程中，随机失活可以被认为是对完整神经网络抽样出一些子集，每次基于输入数据值只更新子网络的参数（然而，数量巨大的子网络们并不是相互独立的，因为它们都共享参数）
- 在测试（或者验证）时候不使用随机失活，可以理解为对数量巨大的子网络们做了模型集成，以此来计算出一个平均的预测
- `rand(*x.shape)`：`np.random.rand()`括号里加的是个int型的数，而a.shape结果并不是一个int型的数，应在a.shape前面加个*号，原因见[python 函数参数的传递(参数带星号的说明)](http://www.cnblogs.com/smiler/archive/2010/08/02/1790132.html)
### Q4: ConvNet on CIFAR-10 (Completed)
- ipython的内核是py35的，所以若缺少模块得到py35环境下安装，调试error时要认真看清楚信息


## Assignment3

### Q1: Image Captioning with Vanilla RNNs (Not Yet)

### Q2: Image Captioning with LSTMs (Not Yet)

### Q3: Image Gradients: Saliency maps and Fooling Images (Not Yet)

### Q4: Image Generation: Classes, Inversion, DeepDream (Not Yet)


