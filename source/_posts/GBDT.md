---
title: GBDT
date: 2018-11-18
updated: 2020-03-04
categories:
    - 机器学习
tags:
    - Ensemble Learning
    - Boosting
    - Loss函数
mathjax: true
---

提升树也是boosting家族的成员，意味着提升树也采用加法模型（基学习器线性组合）和前向分步算法。
参考资料里面把Gradient Boosting，以及GBDT都解释的很透彻！

# Gradient Boosting 算法

Gradient Boosting:

1. 初始化：$$f_0(x) = \arg\min\limits_{\rho}\sum\limits_{i=1}^NL(y_i, \rho)$$

2. for $m=1$ to M:
    a. 计算负梯度：$$\widetilde{y_i} = -\frac{\partial L(y_i, f_{m-1}(x_i))}{\partial f_{m-1}(x_i)}$$, $$i=1,2,\dots,N$$
    b. 通过最小化平方误差，用基学习器$$h_m(x)$$拟合$$\widetilde{y_i}$$，
    $$
    w_m = \arg\min\limits_{w}\sum\limits_{i=1}^N[\widetilde{y_i} - h_m(x_i, w)]^2
    $$
    c. 使用line search确定步长$\rho_m$，使得$L$最小，
    $$
    \rho_m = \arg\min\limits_{\rho}\sum\limits_{i=1}^NL(y_i, f_{m-1}(x_i) + \rho h_m(x_i, w_m))
    $$
    d. $f_m(x) = f_{m-1}(x) + \rho_m h_m(x, w_m)$
    
3. 输出$$f_M(x)$$

# GBDT算法

如果基学习器$$h_m(x)$$是**回归**决策树模型，则称为GBDT。

1. 初始化：$f_0(x) =\arg\min\limits_{\rho}\sum\limits_{i=1}^NL(y_i,\rho)$

2. for m=1 to M:

    a. 计算负梯度：$\widetilde{y_i} = -\frac{\partial L(y_i, f_{m-1}(x_i))}{\partial f_{m-1}(x_i)}, ~~i=1,2,\dots,N$
    
    b. 通过regression tree模型拟合：$\{R_{jm}\}_{j=1}^J=J-terminal~node~tree(\{\widetilde{y_i}, x_i\}_{i=1}^N)$
    
    c. 通过line search确定叶子节点的权重：$\gamma_{jm} = \arg\min\limits_{\gamma}\sum\limits_{x\in R_{jm}} L(y_i, f_{m-1}(x_i)+\gamma)$
    
    d. $f_m(x)=f_{m-1}(x) + \sum\limits_{j=1}^J\gamma_{jm}I(x\in R_{jm})$

3. 输出$f_M(x)$

决策树寻找最优树的过程其实依靠启发式的分裂准则，训练样本是$\{\widetilde{y_i}, x_i\}$，$h_m(x_i,w) = \sum\limits_{j=1}^J b_jI(x_i\in R_{jm})$，
其中$w=(R_{jm},b_{jm})^J_1$，最终将N个样本划分为J个区域：$\{R_{jm}\}, j=1,\dots,J$. 同时求得参数R和b。每个区域为上的样本点都是相同的输出值，即$x\in R_{jm}\longrightarrow h_m(x)=b_{jm}$. 但是b会在下一步与$\rho$结合考虑得到$\gamma_{jm}$，所以b步主要是得到$R_{jm}$.

b步得到了J个区域，即J个叶子节点，那么叶子节点的取值是多少？也就是这棵树到底输出多少？对于不同的损失函数，叶子节点的值也不一样。 第m颗树的第j个叶子节点的值为$$\gamma_{jm}=b_{jm}\rho_m$$。在GBDT里，通常将c步称作Shrinkage。

与Gradient Boosting形式一致的话 ，d步可写成$f_m(x)=f_{m-1}(x) + \rho_m\sum\limits_{j=1}^Jb_{jm}I(x\in R_{jm})$.

 **$f_m(x)$是前m轮的累积求和。**

# 残差，负梯度，损失函数

一方面，对于加法模型，经过迭代，在每一步$m$得到$f_m(x)$，与真实值的差即为残差：$r=y-f_m(x)$.

另一方面，GBDT每一步$m$迭代拟合的是负梯度，是损失函数关于上一轮预测值的负梯度$\frac{\partial L(y_i, f_{m-1}(x_i))}{\partial f_{m-1}(x_i)}$，预测值在负梯度方向上使得损失函数减小。

**问题：**每一步迭代的目的，是缩小残差吗？如果是的话，拟合出来的值$\gamma_{jm}I(x\in R_{jm})$与之前的$$f_{m-1}$$相加$$f_m=f_{m-1}+ Fitted Residual$$，最终拟合预测值$f_M$.

回想泰勒展开，函数的变化量可以展开为各阶导数之和。也就是残差可以表示为各阶导数的线性组合，我们这里只考虑一阶导数，只用一阶导数（负梯度）这一项来近似表示残差。

## 1.square loss
**回归问题**常用的损失函数为square loss $l(y_i,\hat{y}_i)=(y_i-\hat{y}_i)^2$可以很好的解释gradient boosting，拟合的负梯度就是残差，因为拟合的损失函数与泰勒展开近似之后的形式完全一样：
$$\begin{array}{ll}
Loss^{(t)} & = \sum\limits_{i=1}^n [y_i - (\hat{y}_i^{(t-1)}+f_t(x_i))]^2 + \Omega(f_t) + const \\
          & = \sum\limits_{i=1}^n [(y_i-\hat{y}_i^{(t-1)})^2 + 2(\hat{y}_i^{(t-1)}-y_i)f_t(x_i) + f_t(x_i)^2] + \Omega(f_t) + const 
\end{array}$$
回忆泰勒展开式$f(x+\Delta x) \simeq f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2$，所以在迭代后mean square loss损失函数的展开形式与泰勒展开形式完全一样。

## 2. logloss

**二分类问题**常用对数损失函数。

回顾从线性回归到逻辑回归：是将线性拟合的值，经过logistics函数$h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}$，$h_{\theta}(x)$的值表示结果是1的概率。所以$P(y=1|x,\theta)=h_{\theta}(x), P(y=0|x,\theta)=1-h_{\theta}(x)$. 综合起来可写成$P(y|x,\theta)=(h_{\theta}(x))^y(1-h_{\theta}(x))^{1-y}$. 对数似然为$\prod\limits_{i=1}^N(h_{\theta}(x_i))^{y_i}(1-h_{\theta}(x_i))^{1-y_i}$，连乘不方便计算，取对数变成求和，即为对数损失函数。因为似然函数是求最大值时的$\theta$，所以这里取负号求最小化损失函数。
$$
L(\theta)=-\frac{1}{N}\sum\limits_{i=1}^N[y_i\log h_{\theta}(x_i)+(1-y_i)\log(1-h_{\theta}(x_i))]
$$
GBDT用的是回归树来解决分类问题，与逻辑回归的思想有点类似，同样是先线性拟合，然后再求概率。二分类问题的GBDT也常用对数损失函数。

对单个样本$i$，损失函数为$-y_i\log\hat{y}_i-(1-y_i)\log(1-\hat{y}_i)$，其中$\hat{y}_i=h_{\theta}(x)=\frac{1}{1+e^{-f_m(x_i)}}$，其中$f_m(x_i)$是m轮迭代后树模型相加的值。将$\hat{y}_i$替换为$f_m(x)$后，损失函数可写为
$$
L(y_i,f_m(x_i))=y_i\log(1+e^{-f_m(x_i)})+(1-y_i)[f_m(x_i)+\log(1+e^{-f_m(x_i)})]
$$
损失函数的负梯度为
$$
r_{im}=-\bigg|\frac{\partial L(y_i, f_m(x_i))}{\partial f_m(x_i)}\bigg|= y_i-\frac{1}{1+e^{-f_m(x_i)}}=y_i-\hat{y}_i
$$
可以看到，从形式上跟回归问题的负梯度的一致的！

但是对数损失函数在求line serach的时候，$\gamma_{jm} = \arg\min\limits_{\gamma}\sum\limits_{x\in R_{jm}} L(y_i, f_{m-1}(x_i)+\gamma)$没有闭式解，所以用近似值代替（推导见[6-2]）：
$$
\gamma_{jm} = \frac{\sum\limits_{x_i\in R_{jm}}r_{im}}{\sum\limits_{x_i\in R_{jm}}(y_i-r_{im})(1-y_i+r_{im})}
$$
就是用负梯度以及$y_i$来近似代替line search得到的shrinkage $\gamma_{jm}$. 原本负梯度只是用来当作基模型的label，这里还多了一个用途。

另外，初始化的基模型为$f_0(x) = \log\frac{p}{1-p}$，这是因为类似于逻辑回归中的$\theta^Tx=\log\frac{p}{1-p}$，GBDT中的$f(x)$就类似于逻辑回归中的$\theta^Tx$.  二元GBDT分类算法和逻辑回归思想一样，用一系列的梯度提升树去拟合这个对数几率$\log\frac{p}{1-p}$，二分类模型的最终表达式为
$$
P(y=1|x)=\frac{1}{1+e^{-F_M(x)}}
$$

## 3. cross-entropy

**多分类问题**，相较于逻辑回归，假设有K类标签，样本结果为k的概率是
$$
P(y=t|x,\theta) = \frac{e^{\theta_t^Tx}}{\sum\limits_{k=1}^Ke^{\theta_k^Tx}}
$$

即通过softmax将预测结果归一化，如果不是线性模型，用$h_{\theta}(x)$来表示$\theta^Tx$。softmax一般也对应着cross-entropy损失函数（单个样本）
$$
Loss =  -\log \prod\limits_{k=1}^KP(y_k|x)^{y_k}=-\sum\limits_{k=1}^Ky_k\log(h_{\theta_k}(x))
$$
GBDT在处理多分类问题的时候，实际上**每一轮都训练K颗树**，每一类都训练一个0-1二分类的回归树模型（属于第k类的样本标签为1，不属于第k类的标签为0），去拟合softmax的K个分支。

用$f(x)$代替$h_{\theta}(x)$，则**单个样本**的损失函数可写为
$$
Loss(y, f_{1}(x),f_2(x), \dots, f_K(x)) = -\sum\limits_{k=1}^Ky_k\log(h_{\theta_k}(x)) = -\sum\limits_{k=1}^Ky_k\log\frac{e^{f_{k}(x)}}{\sum\limits_{k'=1}^Ke^{f_{k'}(x)}}
$$
这里是单个样本的损失函数，注意$f_k(x)$并非指第k轮迭代，而是在同一轮迭代中第k类分支的树模型。该样本在第k类的负梯度为
$$
-\frac{\partial loss}{\partial f_k(x)}=y_k-\frac{e^{f_k(x)}}{\sum_{k'=1}^Ke^{f_{k'}(x)}} = y_k-\hat{y}_k
$$
这个形式同样与square loss和logloss保持一致，同样认为是在拟合样本真实值与预测值的概率差。

## 4. 总结

前面介绍了三种常见的损失函数square loss，logloss和cross-entropy，分别用于回归问题、二分类和多分类问题。**虽然损失函数的形式不同，GBDT拟合的负梯度，都可以理解成在拟合残差**。只不过square loss就是真实值与预测值之差。而logloss和cross-entropy的残差是指真实值与预测概率之差，每一轮得到的$f(x)$需要经过logistic函数或softmax来变成概率，拟合上一轮概率值的残差：$y_i-\frac{1}{1+e^{-F_{m-1}(x_i)}}$(这里以二分类为例)。分类的残差拟合不容易理解，就是每一轮训练的回归树，是需要经过变换成为概率值来拟合的。

# 用Decision Tree来拟合残差

- 无论是回归问题还是分类问题，都是用回归决策树来拟合
- 决策树可以把样本划分为多个区域，每个区域为上的样本点都是相同的输出值：
    单颗决策树可表示为$$h(x,(R_j,b_j)^J_1)=\sum\limits_{j=1}^J b_jI(x\in R_j)$$
    其中，$$R_j$$为$$J$$个独立区域（即各个叶子结点），$$b_j$$为各区域上的输出值
    于是，2.b可写成$$(R_{jm})_1^J=\arg\min\limits_{R_{jm}}\sum\limits_{i=1}^N[\widetilde{y_i} - h_m(x_i,(R_{jm},b_{jm})^J_1)]^2$$
<!-- - 回归问题的值为该区域样本点的均值，分类问题是众数 -->
- 无论是回归问题还是分类问题，都是用回归树来拟合。
    即使是分类问题，也是使用回归树来拟合，比如使用logloss来拟合，拟合的是概率值。关于logloss的讨论，参考下面两篇
    https://github.com/wcfrank/GBDT/blob/master/2.%20Loss%20functions.ipynb
    https://github.com/wcfrank/GBDT/blob/master/3.%20compute_loss.ipynb
- 2.b得到$$R_{jm}$$，2.c得到$$\gamma_{jm}$$：用回归树来拟合出$$R_{jm}$$，令$$\gamma_{jm}=\rho_mb_{jm}$$，2.b和2.c可以结合写成
$$\gamma_{jm} = \arg\min\limits_{\gamma}\sum\limits_{x_i\in R_{jm}}L(y_i, f_{m-1}(x_i)+\gamma)$$
   - 如果是Squared loss，最优值为region中残差的平均数，$$\gamma_{jm} = \arg\min\limits_{\gamma}\sum\limits_{x_i\in R_{jm}}((y_i-f_{m-1}(x_i))-\gamma)^2 = average_{x_i\in R_{jm}}\widetilde{y_i}$$
   - 如果是Absolute loss，最优值为region中残差的中位数，$$\gamma_{jm} = \arg\min\limits_{\gamma}\sum\limits_{x_i\in R_{jm}}|(y_i-f_{m-1}(x_i))-\gamma| = median_{x_i\in R_{jm}} (y_i - f_{m-1}(x_i))$$
   - 如果是Log loss，$$\gamma_{jm} = \arg\min\limits_{\gamma}\sum\limits_{x_i\in R_{jm}}\log(1+e^{-2y_i(f_{m-1}(x_i)+\gamma)})\approx \frac{\sum\limits_{x_i\in R_{jm}}\widetilde{\widetilde{y_i}}}{\sum\limits_{x_i\in R_{jm}}|\widetilde{y_i}|(2-\widetilde{y_i})}$$，见参考资料5


# 参考资料
1. [Introduction to Boosted Trees (Chen Tianqi)](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
1. [GBDT原理详解](https://www.cnblogs.com/ScorpioLu/p/8296994.html)
1. [Introduction To Gradient Boosting algorithm (simplistic n graphical) - Machine Learning](https://www.youtube.com/watch?v=ErDgauqnTHk)
1. [How to explain gradient boosting](https://explained.ai/gradient-boosting/index.html)
1. [GBDT原理与Sklearn源码分析-回归篇](https://blog.csdn.net/qq_22238533/article/details/79185969)
1. Mictrostrong专栏
   - [深入理解GBDT回归算法](https://zhuanlan.zhihu.com/p/81016622)
   - [深入理解GBDT二分类算法](https://zhuanlan.zhihu.com/p/89549390)
   - [深入理解GBDT多分类算法](https://zhuanlan.zhihu.com/p/91652813)

7. [GBDT算法用于分类问题](https://zhuanlan.zhihu.com/p/46445201)