---
title: 梯度爆炸&梯度弥散
date: 2019-03-17
categories:
    - 深度学习
tags:
    - 梯度
mathjax: true
---

# 梯度爆炸&梯度弥散

梯度爆炸和梯度弥散都是由训练时的反向传播引起的。

![Graident Vanishing](https://images2017.cnblogs.com/blog/1081032/201709/1081032-20170905102905460-307678090.png)

上图是做预测时，前向传播梯度的影响；训练时是反向传播，从后向前，前面层的梯度越来越小。

靠近输出层的hidden layer 梯度大，参数更新快，所以很快就会收敛；而靠近输入层的hidden layer 梯度小，参数更新慢，几乎就和初始状态一样，随机分布。这种现象就是梯度弥散。而在另一种情况中，前面layer的梯度通过训练变大，而后面layer的梯度指数级增大，这种现象又叫做梯度爆炸。总的来说，就是在这个深度网络中，梯度相当不稳定(unstable)。[2]




![Gradient Vanishing](https://images2017.cnblogs.com/blog/1081032/201709/1081032-20170905103102663-500891744.png)


根据sigmoid的特点，它会将-∞～+∞之间的输入压缩到0～1之间。当input的值更新时，output会有很小的更新。又因为上一层的输出将作为后一层的输入，而输出经过sigmoid后更新速率会逐步衰减，直到输出层只会有微乎其微的更新。[2]

### 梯度解释

![gradient](https://images2017.cnblogs.com/blog/1081032/201709/1081032-20170905103703554-551903486.png)

每一层只有一个神经元，我们分析参数bias的变化。

符号：第一层为输入，中间三个隐藏层，然后输出，C为损失函数。输入为$$a_0​$$，输出为$$a_4​$$，每一层有 $$z=w*a+b​$$. 上一层的输出为下一层的输入，需要经过线性组合+激活函数$$\sigma​$$，变成下一层的输出：例如$$a_1=\sigma(z_1)=\sigma(w_1a_0+b_1)​$$

$$\begin{array}{rl}
\frac{\partial C}{\partial b_1}= & \frac{\partial C}{\partial a_4} \frac{\partial a_4}{\partial z_4}\frac{\partial z_4}{\partial a_3}\frac{\partial a_3}{\partial z_3}\frac{\partial z_3}{\partial a_2}\frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial a_1}\frac{\partial a_1}{\partial z_1}\frac{\partial z_1}{\partial b_1}\\
= & \frac{\partial C}{\partial a_4}\sigma'(z_4) w_4 \sigma'(z_3) w_3 \sigma'(z_2) w_2 \sigma'(z_1) * 1
\end{array}​$$

sigmoid导数在0取得最大值1/4；如果我们使用均值为0，方差为1的高斯分布初始化参数w，有|w| < 1，可以看出随着网络层数的加深$$|w_j\sigma'(z_j)|<\frac{1}{4}$$的term也会变多，最后的乘积会指数级衰减，这就是梯度弥散的根本原因。如果w取值很大，这里四个w的乘积就很大，梯度爆炸。

梯度爆炸相对容易解决，通过裁剪后的优化算法即可解决，如gradient clipping（如果梯度的范围大于某给定值，将梯度同比例收缩，或者将大于threshold的梯度设为threshold值）





# 解决方案

### 使用relu、leakrelu、elu等激活函数

![relu](https://pic4.zhimg.com/80/v2-f52ca25ffd6829ee2dfd849c256119b3_hd.jpg)

Relu激活函数在x>0的时候，导数恒为1，不存在梯度爆炸、梯度弥散的问题了，每层的网络都可以得到相同的更新速度。

Relu的优点：解决了梯度爆炸、梯度弥散的问题；计算方便，速度快；加速网络训练

Relu的缺点：复数部分恒为0，会导致一些神经元无法激活（可通过设置小学习率部分解决）；输出不是以0为中心的

### Batch Normalization：

目前已经被广泛的应用到了各大网络中，具有加速网络收敛速度，提升训练稳定性的效果，Batchnorm本质上是解决反向传播过程中的梯度问题。通过规范化操作将输出信号x规范化到均值为0，方差为1保证网络的稳定性。

batchnorm解决梯度的问题上。具体来说就是反向传播中，经过每一层的梯度会乘以该层的权重，$$f_3 = f_2(w^T*x+b)$$，那么反向传播中，$$\frac{\partial f_2}{\partial x} = \frac{\partial f_2}{\partial f_1}w$$. 反向传播式子中有w的存在，所以w 的大小影响了梯度的消失和爆炸，batchnorm就是通过对每一层的输出做scale和shift的方法，通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到接近均值为0方差为1的标准正太分布，即严重偏离的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，使得让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。

### 残差结构：

CNN模型的梯度弥散，用残差模型ResNet改进 



### LSTM：

RNN模型的梯度弥散，用LSTM或者GRU来改进



参考资料

1. [详解深度学习中的梯度消失、爆炸原因及其解决方法](https://zhuanlan.zhihu.com/p/33006526)
2. [梯度弥散与梯度爆炸](https://www.cnblogs.com/yangmang/p/7477802.html)
3. [LSTM如何来避免梯度弥散和梯度爆炸？](https://www.zhihu.com/question/34878706)