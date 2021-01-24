---
title: GCN
date: 2021-01-20
updated: 2021-01-20
categories:
    - GNN
tags:
    - GCN
    - Fourier Transform
    - Laplace
mathjax: true
---

前面两篇文章分别介绍了[傅立叶变换](Fourier_transform.md)和[图的拉普拉斯矩阵](Laplace.md)，有这两篇文章作为基础，我们终于可以学习GCN了。

GCN的理论基础是Spectual Graph Theory，Spectual Graph Theory是将图的拉普拉斯矩阵进行谱分解，用它的特征值和特征向量来表示图上的信息。这样图上的信息就由vertex domain变换到了spectual domain上，spectual domain与顶点信息没有关系了，而是考虑每个特征向量的“份量”。那么拉普拉斯的特征值和特征向量与傅立叶变换有什么关系呢？本文首先讨论的图傅立叶变换，就会将前面两篇文章的知识点结合起来。

# Graph Fourier Transform

在傅立叶变换一文中，最后的总结：

> 傅立叶逆变换本质是把任意一个函数表示成了若干个正交基函数的线性组合；
> 傅立叶正变换本质是求线性组合的系数。也就是由原函数和基函数的共轭的内积求得。

比如在传统的傅立叶变换中，就是用不同频率的三角函数作为一组正交基，它们的线性组合来表示时域信号。那图上的信号由什么来表示呢？

## 图傅立叶正变换

对于图上的信号，也可以进行傅立叶变换，即找到一组正交基的线性组合来表示图的信号。那么问题就是图的信号由一组什么样的基来组成呢？

回忆传统的傅立叶变换$g(\hat{f}) =\sum\limits_{t=-\infty}^\infty F(t)e^{-2\pi i\hat{f}t}= \int_{-\infty}^\infty F(t)e^{-2\pi i \hat{f}t}dt$，令$w=2\pi\hat{f}$，那么傅立叶变换的公式为
$$
g(w) = \int F(t)e^{-iwt}dt
$$

即时域信号可以由一组不同频率的基函数$e^{-iwt}$的线性组合来表示。我们来观察一下这一组基$e^{-iwt}$，除了它是正交基，还满足是拉普拉斯算子的特征函数。为什么呢？

首先我们把特征向量推广到特征函数：特征向量满足$Ax=\lambda x$，把矩阵$A$推广为任意的一种变换（算子），把特征向量$x$推广为特征函数（无限维特征向量）。我们要验证基函数$e^{-iwt}$是拉普拉斯算子$\Delta=\sum\limits_{i}\frac{\partial^2}{\partial^2 x_i}$的特征函数，即
$$
\Delta e^{-iwt} = \frac{\partial^2}{\partial^2 t}e^{-iwt} = -w^2e^{-iwt}
$$
所以**传统傅立叶变换的基$e^{-iwt}$是拉普拉斯算子的特征函数，傅立叶变换是时域信号与拉普拉斯算子特征函数的积分。**由之前的[文章](Laplace.md)，图上的拉普拉斯算子就是图的拉普拉斯矩阵，所以把时域信号推广到**离散的**图信号上，**图上的傅立叶变换是图信号与拉普拉斯矩阵特征向量的求和。**
$$
g(\lambda_l) = \sum\limits_{i=1}^n f(i)*u^*_l(i)
$$
以上就是图的傅立叶变换。其中，图有n个顶点，顶点$i$上的图信号为$f(i)$（图信号对应传统傅立叶变换的时域信号）；图上拉普拉斯矩阵的n个特征向量为$u_1,u_2,\dots,u_n$，而且这n个特征向量是正交的，第$l$个特征向量的第$i$个分量是$u_l(i)$；$\lambda_l$为图拉普拉斯矩阵的第$l$个特征向量对应的特征值。上面的求和运算使用的是特征向量的共轭$u^*_l$，这是因为特征向量是定义在复平面的，为了保证正交性，所以内积用共轭，这在[傅立叶变换](Fourier_transform.md)一文中解释过。

将图上的傅立叶变换写成矩阵形式：
$$
\begin{pmatrix}g(\lambda_1)\\g(\lambda_2)\\\dots\\g(\lambda_n)\end{pmatrix}=
\begin{pmatrix}u_1(1)&u_1(2)&\dots&u_1(n)\\
u_2(1)&u_2(2)&\dots&u_2(n)\\
\vdots&\vdots&\ddots&\vdots\\
u_n(1)&u_n(2)&\dots&u_n(n)
\end{pmatrix}
\begin{pmatrix}
f(1)\\f(2)\\\dots\\f(n)
\end{pmatrix}
$$
简写为$g=U^Tf$. 这里的$U$就是[拉普拉斯矩阵](Laplace.md)一文中的$U$，即

> $U=(u_1,u_2,\dots,u_n)$是**列向量**为单位特征向量的矩阵，且$U$是正交矩阵，即$UU^T=I$

## 图傅立叶逆变换

类似的，传统傅立叶逆变换公式$F(t) = \sum\limits_{\hat{f}=-\infty}^\infty g(\hat{f})e^{2\pi i\hat{f}t}=\int_{-\infty}^\infty g(\hat{f})e^{2\pi i\hat{f}t}d\hat{f}$，令$w=2\pi\hat{f}$，则傅立叶逆变换公式为$F(t)=\frac{1}{2\pi}\int_{w}g(w)e^{iwt}dw$，从频域变换到时域。在图傅立叶逆变换中，从频域信号变换到图上的信号，
$$
f(i) = \sum\limits_{l=1}^n g(\lambda_l)u_l(i)
$$
将图上的傅立叶逆变换写成矩阵形式：
$$
\begin{pmatrix}
f(1)\\f(2)\\\dots\\f(n)
\end{pmatrix}=
\begin{pmatrix}u_1(1)&u_2(1)&\dots&u_n(1)\\
u_1(2)&u_2(2)&\dots&u_n(2)\\
\vdots&\vdots&\ddots&\vdots\\
u_1(n)&u_2(n)&\dots&u_n(n)
\end{pmatrix}
\begin{pmatrix}g(\lambda_1)\\g(\lambda_2)\\\dots\\g(\lambda_n)\end{pmatrix}
$$
简写为$f=UU^{-1}f=UU^Tf=Ug$. 

## 两点疑问revisited

1. 为什么拉普拉斯矩阵的**特征向量**可以作为图傅立叶变换的基？
傅立叶变换的本质是把一个函数表示称一组基函数的线性组合。通过前面的公式看出，图上的信号$f(i)$，其中$i$是顶点，可以表示成拉普拉斯矩阵特征向量的线性组合。因为拉普拉斯矩阵的特征向量$u_1,u_2,\dots,u_n$是n维空间（n个顶点）下n个线性无关的正交向量，所以这组特征向量构成空间的一组基，可以表示**任意**信号$f(i)$. 另外一个[解释](https://www.zhihu.com/question/355874991/answer/1204283923)的角度比较难懂，姑且看之。
2. 为什么拉普拉斯矩阵的**特征值**表示的频率是什么？
在图上确实无法直观的展示“频率”这个概念。至少经过图傅立叶变换，顶点的信号被变换成了“频率”信号，从此我们在**spectual domain**上研究问题，而不是在**vertex domain**上了。将拉普拉斯的n个特征值从小到大排列，$\lambda_1\le\lambda_2\le\dots\lambda_n$，因为拉普拉斯矩阵是半正定的，所以这些特征值都非负。而且最小的特征值$\lambda_1=0$，这是因为全1向量也是拉普拉斯矩阵的特征向量（拉普拉斯矩阵的定义），对应的特征值为0。在图的n维空间中，越小的特征值$\lambda_l$代表着特征向量$u_l$的占有的份量越少，是可以忽略的低频部分。这个跟图像压缩、PCA降维有异曲同工之妙，都是把特征值小的部分变成0，保留特征值大的部分。

# 图卷积

## 函数卷积定义

关于卷积的定义可以参考[此回答](https://www.zhihu.com/question/22298352/answer/637156871)，此问题下许多其他回答都蛮好。教科书上的定义函数$f_1, f_2$的卷积为$f_1*f_2(n)$，

- 连续形式：$f_1*f_2(n)=\int_{-\infty}^{\infty}f_1(\tau)f_2(n-\tau)d\tau$
- 离散形式：$(f_1*f_2)(n) = \sum\limits_{\tau=-\infty}^{\infty}f_1(\tau)f_2(n-\tau)$

形象的解释就是把函数$g$进行翻转，把函数$g$沿着数轴从右边对折，到了左边，也就是“**卷**”的由来。然后把函数沿着自变量方向平移$n$，再对两个函数的对应元素相乘，再求和，这就是“**积**”的由来。

## 函数卷积的傅里叶变换

传统傅里叶卷积：**函数卷积的傅里叶变换是函数傅里叶变换的乘积；或者说函数卷积是函数傅里叶变化乘积的逆变换。**

设两个时域信号$f_1(t), f_2(t)$，它们卷积的傅里叶变换为$f_1(t)*f_2(t)=\int_{-\infty}^\infty f_1(\tau)f_2(t-\tau)d\tau$，下面证明卷积的傅里叶变换是傅里叶变换的乘积：
$$
\begin{array}{rl}
\text{Fourier}(f_1(t)*f_2(t))&=\int_{-\infty}^\infty[\int_{-\infty}^\infty f_1(\tau)f_2(t-\tau)d\tau]e^{-iwt}dt\\
&=\int_{-\infty}^\infty f_1(\tau)e^{-iw\tau}d\tau \int_{-\infty}^\infty f_2(t-\tau)e^{-iw(t-\tau)}d(t-r)\\
&=g_1(w)g_2(w)\\
f_1(t)*f_2(t) &= \text{Inverse_Fourier}(g_1(w)g_2(w))\\
&=\frac{1}{2\pi}\int g_1(w)g_2(w)e^{iwt}dw
\end{array}
$$

## 图卷积

GCN中的C就是convolution，终于到这里了！图的卷积需要类比传统卷积公式并结合傅里叶变换来定义：两个图信号的卷积是图信号傅里叶变换乘积的逆变换。

假设图信号$f$的傅里叶变换为$g=U^Tf$，卷积核（另一个图信号）$h$的傅里叶变换定义为一个对角矩阵
$$
\begin{pmatrix}
\hat{h}(\lambda_1)&&\\
&\ddots&\\
&&\hat{h}(\lambda_n)
\end{pmatrix}
$$
其中$\hat{h}(\lambda_l) = \sum\limits_{i=1}^nh(i)u^*_l(i)$，是**根据需要设计出来的卷积核$h$在图上的傅里叶变换**，其实我们只在乎傅里叶变换之后的矩阵，不在乎原来的卷积核$h$长什么样子。

这两个傅里叶变换的乘积为（这里的$U$与上一章的$U$相同）
$$
\begin{pmatrix}
\hat{h}(\lambda_1)&&\\
&\ddots&\\
&&\hat{h}(\lambda_n)
\end{pmatrix}
U^Tf
$$
最后的，参考图傅里叶逆变换，这个乘积的逆变换为
$$
(f*h)_G=U
\begin{pmatrix}
\hat{h}(\lambda_1)&&\\
&\ddots&\\
&&\hat{h}(\lambda_n)
\end{pmatrix}
U^Tf
$$
以上即为图卷积的公式，也可以写作$(f*h)_G=U((U^Th)\odot(U^Tf))$，其中$\odot$表示Hadamard product，是逐元素乘积运算，[这篇文章](https://zhuanlan.zhihu.com/p/121090537)证明了以上两个公式是等价的。

# 深度学习中的图卷积

CNN中的卷积核是可训练的、参数共享的kernel，在图神经网络中也是一样的目的。结合上面图卷积的结论，**这个共享的卷积参数就是对角矩阵$\text{diag}(\hat{h}(\lambda_l))$.** 

我们只讨论一层图卷积神经网络的结构$y_=\sigma(U\cdot\text{diag}(\hat{h}(\lambda_l)\cdot U^Tx)$，其中$x$是图信号（每个顶点的feature vector），$\sigma$是激活函数，$y$是经过这层图卷积网络的输出。

## GCN v1

把卷积核$\text{diag}(\hat{h}(\lambda_l))$简化为$\text{diag}(\theta_l)$（或者记为$g_{\theta}(\Lambda)$），这样$y_=\sigma(U\cdot g_{\theta}(\Lambda)\cdot U^Tx)$，其中
$$
g_{\theta}(\Lambda)=
\begin{pmatrix}
\theta_1&&\\
&\ddots&\\
&&\theta_n
\end{pmatrix}
$$
矩阵中的元素$(\theta_1,\theta_2,\dots,\theta_n)$就是神经网络里面的weights参数，用来训练。

这种GCN的缺点：

- 每一次前向传播都需要计算$(U\cdot\text{diag}(\hat{h}(\lambda_l)\cdot U^T)$三个矩阵乘积，时间复杂度太高
- 这种卷积核不具有Spatial localization
- 卷积核需要n个参数

## GCN v2

基于v1的几个缺点进行改进，出现了第二代卷积核。

把卷积核$\text{diag}(\hat{h}(\lambda_l))$的每一项设为$\sum\limits_{j=0}^K\alpha_j\lambda_l^j$，则
$$
g_{\theta}(\Lambda)=
\begin{pmatrix}
\sum\limits_{j=0}^K\alpha_j\lambda_1^j&&\\
&\ddots&\\
&&\sum\limits_{j=0}^K\alpha_j\lambda_n^j
\end{pmatrix}
$$

# Reference

0. [如何理解 Graph Convolutional Network（GCN）？](https://www.zhihu.com/question/54504471/answer/332657604) （重点参考）

1. Graph Fourier Transform
	- [Wiki: Graph Fourier Transform](https://en.wikipedia.org/wiki/Graph_Fourier_Transform)
	- [图卷积神经网络系列：2. | 图傅里叶变换](https://zhuanlan.zhihu.com/p/137897522)
	- [图上的傅里叶变换和逆变换](https://zhuanlan.zhihu.com/p/272559663)（重点参考）
2. 图卷积
   - [如何通俗易懂地解释卷积？](https://www.zhihu.com/question/22298352)
   - [从传统傅里叶变换到图卷积](https://zhuanlan.zhihu.com/p/123362731)