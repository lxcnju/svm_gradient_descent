# svm_gradient_descent
Using neural networks to solve svm, including linear and kernel type.

利用神经网络和梯度下降来求解SVM，包括线性SVM和采用核技巧的SVM。神经网络的搭建采用Python3.6 + Tensorflow1.3实现。

* 代码架构
* 原理解析
  * Linear SVM
  * Kernel SVM
  
## 代码架构
 * linear_svm_gradient.py 线性SVM利用神经网络实现
 * kernel_svm_gradient.py 核技巧SVM利用神经网络实现
 * data 实验数据集，采用[svmguide1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#svmguide1)，二分类，训练集(3089,4)，测试集(4000,4)
 * pics 实验得到的结果图片

## 原理解析
 * Linear SVM <br>
 Linear SVM很简单，利用Tensorflow直接搭建网络即可。Linear SVM的数学公式见下图：<br>
 ![Linear SVM formula](https://github.com/lxcnju/svm_gradient_descent/blob/master/linear_svm.png) <br>
 通过上面公式可以看出，对于一个输入样本，经过权重w和偏置b的作用之后即可得到预测值，然后根据样本的实际标签计算hinge loss，再加上正则化项即可。因此利用神经网络搭建非常方便，见代码linear_svm_gradient.py。 <br>
 搭建好网络之后，开始训练，训练参数主要包括：batch_size，learning_rate，regularizer。其中学习率采用指数下降，每隔100轮衰减为之前的0.95大小，下面给出一些训练结果。<br>
 当batch_size=8，learning_rate=0.001，regularizer=0.0001时，绘制出训练过程模型在训练集和测试集上的损失和准确率如下图，可见训练过程不稳定，波动很大。<br>
 ![Linear SVM Loss 1](https://github.com/lxcnju/svm_gradient_descent/blob/master/loss_1.png)
 ![Linear SVM Accuracy 1](https://github.com/lxcnju/svm_gradient_descent/blob/master/accu_1.png) <br>
 当batch_size=32，learning_rate=0.001，regularizer=0.0001时，绘制出训练过程模型在训练集和测试集上的损失和准确率如下图，模型有过拟合现象，在训练轮数超过300时，训练集上损失继续减小，但是测试集上损失开始增大。<br>
 ![Linear SVM Loss 2](https://github.com/lxcnju/svm_gradient_descent/blob/master/loss_2.png)
 ![Linear SVM Accuracy 2](https://github.com/lxcnju/svm_gradient_descent/blob/master/accu_2.png) <br>
 当batch_size=32，learning_rate=0.001，regularizer=0.001时，绘制出训练过程模型在训练集和测试集上的损失和准确率如下图，模型训练过程比较好，最终测试集上准确率可以达到92.5%。<br>
 ![Linear SVM Loss 3](https://github.com/lxcnju/svm_gradient_descent/blob/master/loss_3.png)
 ![Linear SVM Accuracy 3](https://github.com/lxcnju/svm_gradient_descent/blob/master/accu_3.png) <br>
 
 * Kernel SVM <br>
 Kernel SVM是先将数据点从低维空间映射到高维空间，然后再利用Linear SVM进行求解。但是如何找到低维空间到高维空间的映射函数则是一个很复杂的过程，并且有时高维空间是无穷维，则更找不到。所以，引入了核技巧，高维空间的两个数据点的内积可以等价为低维空间的内积经过核函数处理。所以采用核技巧的SVM的数学公式如下：<br>
 ![Kernel SVM formula](https://github.com/lxcnju/svm_gradient_descent/blob/master/kernel_svm.png) <br>
 从上面可以看出，对于一个样本，要经过和所有训练数据进行内积操作，然后经过核函数处理，再线性加权得到。因此，这时搭建网络需要采用一个隐层大小为训练中数据样本个数的网络层，激活函数为Kernel Function，比如tanh和radial等，本实验采用tanh激活函数来实现。代码详见kernel_svm_gradient.py。 <br>
 ![kernel SVM Loss](https://github.com/lxcnju/svm_gradient_descent/blob/master/kernel_loss_1.png)
 ![kernel SVM Accuracy](https://github.com/lxcnju/svm_gradient_descent/blob/master/kernel_accu_1.png) <br>
 可以看出训练过程基本上是比较稳定的，但是仍有小的波动，从训练损失看，没有出现过拟合现象，最终测试集准确率可以达到90%。
