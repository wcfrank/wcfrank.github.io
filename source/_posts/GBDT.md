---
title: GBDT
date: 2018-11-18
updated: 2019-01-13
categories:
    - 机器学习
tags:
    - Ensemble Learning
    - Boosting
mathjax: true
---

提升树也是boosting家族的成员，意味着提升树也采用加法模型（基学习器线性组合）和前向分步算法。
参考资料里面把Gradient Boosting，以及GBDT都解释的很透彻！

# Gradient Boosting 算法
1. 初始化：$$f_0(x) = \arg\min\limits_{\alpha}\sum\limits_{i=1}^NL(y_i, \alpha)$$
2. for $m=1$ to M:
    a. 计算负梯度：$$\widetilde{y_i} = -\frac{\partial L(y_i, f_{m-1}(x_i))}{\partial f_{m-1}(x_i)}$$, $$i=1,2,\dots,N$$
    b. 通过最小化平方误差，用基学习器$$h_m(x)$$拟合$$\widetilde{y_i}$$，
    $$w_m = \arg\min\limits_{w}\sum\limits_{i=1}^N[\widetilde{y_i} - h_m(x_i, w)]^2$$
    c. 使用line search确定步长$\rho_m$，使得$L$最小，
    $$\rho_m = \arg\min\limits_{\rho}\sum\limits_{i=1}^NL(y_i, f_{m-1}(x_i) + \rho h_m(x_i, w_m))$$
    d. $$f_m(x) = f_{m-1}(x) + \rho_m h_m(x, w_m)$$
3. 输出$$f_M(x)$$

如果基学习器$$h_m(x)$$是决策树模型，则称为GBDT。再次概括一下Gradient Boosting里面迭代的思路：（参考资料5）
a. 求梯度
b. 用基学习器来拟合梯度：通过极小化mean squared error，确定基学习器的参数，得到基学习器$$tree_m(x,w)$$。对于决策树，寻找一个最优的树的过程其实依靠的就是启发式的分裂准则，训练样本是$$\{\widetilde{y_i}, x_i\}$$，
$$h_m(x_i,w) = \sum\limits_{j=1}^J b_jI(x_i\in R_{jm})$$
其中$$w=(R_{jm},b_{jm})^J_1$$，最终将N个样本划分为J个区域：$$\{R_{jm}\}, j=1,\dots,J$$. 同时求得参数R和b。每个区域为上的样本点都是相同的输出值，即$$x\in R_{jm}\longrightarrow h_m(x)=b_{jm}$$.但是b会在下一步与$$\rho$$结合考虑得到$$\gamma_{jm}$$，所以这一步主要是得到$$R_{jm}$$.
c. 上一步得到了J个区域，即J个叶子节点，那么叶子节点的取值是多少？也就是这棵树到底输出多少？对于不同的损失函数，叶子节点的值也不一样。 第m颗树的第j个叶子节点的值为$$\gamma_{jm}=b_{jm}\rho_m$$。
c. 在GBDT里，通常将这个过程作为Shrinkage，也就是把$$\rho_m$$做为学习率。
d. $$f_m(x) = f_{m-1}(x) + \sum\limits_{j=1}^J\gamma_{jm}I(x\in R_{jm})$$

# 解释
对于加法模型，经过迭代，在每一步$m$得到$f_m(x)$，与真实值的差即为残差：$r=y-f_m(x)$.

每一步迭代的目的，是缩小残差。因为是加法模型，每一步$m$迭代拟合的是残差，然后与之前的$$f_{m-1}$$相加$$f_m=f_{m-1}+ Fitted Residual$$，最终拟合真实值y.

**问题：**这个残差怎么拟合呢？
回想泰勒展开，函数的变化量可以展开为各阶导数之和。也就是残差可以表示为各阶导数的线性组合，我们这里只考虑一阶导数，即用一阶导数（负梯度）来拟合残差。

## 1.square loss
损失函数为square loss $l(y_i,\hat{y}_i)=(y_i-\hat{y}_i)^2$可以非常好的解释gradient boosting，拟合的就是残差，这是因为拟合的损失函数与泰勒展开近似之后的形式完全一样：
对square loss，
$$\begin{array}{ll}
Obj^{(t)} & = \sum\limits_{i=1}^n [y_i - (\hat{y}_i^{(t-1)}+f_t(x_i))]^2 + \Omega(f_t) + const \\
          & = \sum\limits_{i=1}^n [(y_i-\hat{y}_i^{(t-1)})^2 + 2(\hat{y}_i^{(t-1)}-y_i)f_t(x_i) + f_t(x_i)^2] + \Omega(f_t) + const 
\end{array}$$
回忆泰勒展开式$f(x+\Delta x) \simeq f(x) + f'(x)\Delta x + \frac{1}{2}f"(x)\Delta x^2$。
令$$g_i = \partial_{\hat{y}^{(t-1)}}l(y_i, \hat{y}^{(t-1)}), h_i = \partial_{\hat{y}^{(t-1)}}^2l(y_i, \hat{y}^{(t-1)})$$
所以损失函数近似为
$$Obj^{(t)}\simeq\sum\limits_{i=1}^n [l(y_i,\hat{y}_i^{(t-1)}) + g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)] + \Omega(f_t) + const$$
对于square loss，它的三次以上的导数均为0，$$g_i = \partial_{\hat{y}^{(t-1)}}(y_i - \hat{y}^{(t-1)})^2 = 2(\hat{y}^{(t-1)} - y_i)$$，$$h_i = \partial_{\hat{y}^{(t-1)}}^2 (\hat{y}^{(t-1)} - y_i)^2 = 2$$，所以它的二次展开式完全等于$f(x+\Delta x)$，不仅仅是近似。当损失函数为square loss时：
$$\begin{array}{ll}
Obj^{(t)} & = \sum\limits_{i=1}^n l(y_i, \hat{y}_i^{(t-1)}+f_t(x_i)) + \Omega(f_t) + const\\  
        & = \sum\limits_{i=1}^n [(y_i-\hat{y}_i^{(t-1)})^2 + 2(\hat{y}_i^{(t-1)}-y_i)f_t(x_i) + f_t(x_i)^2] + \Omega(f_t) + const \\
            & = \sum\limits_{i=1}^n [l(y_i,\hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_2)^2] + \Omega(f_t) + const \\
            & = Obj^{(t)}
\end{array}$$

## 2.用Decision Tree来拟合残差
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
1. (__重点推荐__)[Introduction to Boosted Trees (Chen Tianqi)](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
1. [GBDT原理详解](https://www.cnblogs.com/ScorpioLu/p/8296994.html)
1. [Introduction To Gradient Boosting algorithm (simplistic n graphical) - Machine Learning](https://www.youtube.com/watch?v=ErDgauqnTHk)
1. [How to explain gradient boosting](https://explained.ai/gradient-boosting/index.html)
1. (__重点推荐__)[GBDT原理与Sklearn源码分析-回归篇](https://blog.csdn.net/qq_22238533/article/details/79185969)