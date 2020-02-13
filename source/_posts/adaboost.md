---
title: Adaboost
date: 2018-11-04
updated: 2018-11-18
categories:
    - 机器学习
tags:
    - Ensemble Learning
    - Boosting
mathjax: true
---

讲到Ensemble Learning的boosting，肯定会讲Adaboost，它是boosting思想最重要的算法之一，另外一个是GBDT。本文主要内容是参考《李书》和一篇博客（见参考资料1，2）.

# Adaboost算法
二分类问题，训练数据集{$(x_1,y_1), (x_2,y_2),\dots,(x_N,y_N)$}，样本的label为$y_i=1$或者-1.

__Adaboost__
> 输入：训练数据集{$T = (x_1,y_1), (x_2,y_2),\dots,(x_N,y_N)$}；和某个弱学习算法（又称学习器、分类器）
> 输出：最终分类器$G(x)$

1. 初始化训练样本的权重分布：$$D_1 = (w_{11},w_{12},\dots,w_{1N}), w_{1i}=\frac{1}{N}$$
1. 对每个学习器$m=1,2,\dots,M$:
    1. 将训练数据集$T$和训练数据的权重$D_m$，带入分类器学习，得到$G_m(x)$
    2. 计算$$G_m(x)$$在训练集上的误分类比率：
    $$e_m = P(G_m(x_i)\neq y_i) = \sum\limits_{i=1}^N w_{mi}I(G_m(x_i)\neq y_i)$$
    3. 计算分类器$G_m(x)$的系数：
    $$\alpha_m = \frac{1}{2}\log\frac{1-e_m}{e_m}$$
    4. 更新训练数据集的权重分布：
    $$D_{m+1} = (w_{m+1,1},w_{m+1,2},\dots,w_{m+1,N})$$
    $$w_{m+1,i} = \frac{w_{mi}}{Z_m}e^{-\alpha_m y_i G_m(x_i)}, ~~~i = 1,2,\dots,N$$
    其中$$Z_m = \sum\limits_{i=1}^Nw_{mi}e^{-\alpha_m y_i G_m(x_i)}$$
1. 构建基本分类器的线性组合
$$f(x) = \sum\limits_{m=1}^M\alpha_mG_m(x)$$
并得到最终的分类器：
$$G(x) = sign(f(x)) = sign(\sum\limits_{m=1}^M\alpha_mG_m(x))$$

# 算法的解释
- 第1步，初始化样本的权重为均匀分布，对于第一个学习器，样本的权重相同。即第一个学习器是对原始数据集合上训练。
- 第2.1步，分类器训练的目标就是最小化误分类率$e_m$，
    所以，第2.1步和第2.2步是同时进行的。
    
    > Adaboost算法的最基本性质是它能在学习过程中不断减小训练误差，即误分类比率。（参考资料1-P142-定理8.1，8.2）
- 第2.2步，误分类比率$$e_m$$是被$$G_m(x)$$误分类的样本的权重之和。在第$m$个学习器上的误分类比率：$$e_m = P(G_m(x_i)\neq y_i) = \sum\limits_{G_m(x_i)\neq y_i}w_{mi}$$。（用概率表示是因为第2.4步，每一轮的样本权重之和均为1：$$\sum\limits_{i=1}^Nw_{mi}=1$$. ）
- 第2.2步，默认的弱学习器叫做Decision stump，即只有一层的决策树，只考虑一个特征（一个维度）。选择$e_m$的方式：**遍历每个特征的每个分割方式**，得到的误分类样本权重和最小的为$e_m$。
- 第2.3步，学习器的系数$$\alpha_m$$代表这个学习器的重要性。当$e_m\leq\frac{1}{2}$（即误分类比率低）的时候，有$\alpha_m\geq 0$，并且$e_m$越小$\alpha_m$越大。所以误分类比率越小的分类器，在最终分类器中作用越大。
- 第2.3步，在《西瓜书》里这一步之前，会加一个判断：如果得到$e_m>0.5$，即遍历了每个特征的每种分割方式都找不到合适的弱学习器，跳出迭代，算法终止。
- 第2.4步，权重更新的时候除以$$Z_m$$，这样权重分布$$D_{m+1}$$之和为1，称为一个概率分布。
- 第2.4步，权重的更新可以写成下面的式子
$$w_{m+1,i} = 
\begin{cases}
\frac{w_{mi}}{Z_m}e^{-\alpha_m},  & G_m(x_i)=y_i \\
\frac{w_{mi}}{Z_m}e^{\alpha_m},  & G_m(x_i)\neq y_i
\end{cases}$$
如果$\alpha_m>0$是一个正面作用的分类器，先忽略$Z_m$，对那些分类正确的样本，权重值减小；分类错误的样本呢，权重值变大。这样误分类样本点在下一轮学习器会更加重要。
> 不改变所给的训练数据，只不断改变训练数据的权重分布，使得训练数据在基本分类器的训练中起不同作用，这是Adaboost的特点

# Adaboost算法的数学解释
Adaboost算法是__模型为加法模型的__、__损失函数为指数函数的__、__学习方法为前向分布算法__的二分类问题。
- 加法模型：$f(x) = \sum\limits_{m=1}^M \alpha_m g_m(x,\gamma_m)$. 其中$g_m$是基函数，$\gamma_m$是基函数的参数，$\alpha_m$是基函数的系数
- 指数损失函数：$L(y, f(x)) = e^{-yf(x)}$
    (回忆SVM中的hinge损失函数$L(y,f(x)) = \max(0, 1-yf(x))$)
- 前向分布算法的思想：对于加法模型，从前往后，每一轮只学习一个基函数及其系数，逐步逼近优化损失函数。
    即每一轮$m$的目标就是极小化损失函数
    $$(\alpha_m,g_m) = \arg\min\limits_{\alpha,g}\sum\limits_{i=1}^NL(y_i,f_{m-1}(x_i)+\alpha_m g_m(x_i))$$
    

__应用到Adaboost：__
（加法模型）第$m$轮迭代得到的$$\alpha_m$$，$$G_m(x)$$和$$f_m(x)$$，有
$$f_m(x) = f_{m-1}(x) + \alpha_m G_m(x)$$
（指数损失）又因为损失函数为$L(y,f(x)) = e^{-yf(x)}$，
（前向分布）所以第$m$轮迭代的目标是：要找到能够使损失函数最小化的$\alpha_m$和$G_m(x)$，
$$(\alpha_m, G_m) = \arg\min\limits_{\alpha, G}\sum\limits_{i=1}^Ne^{-y_i(f_{m-1}(x_i) + \alpha G(x_i))}$$
上式可写成
$$(\alpha_m, G_m) = \arg\min\limits_{\alpha, G}\sum\limits_{i=1}^N \bar{w}_{mi} e^{-y_i\alpha G(x_i)} \tag{1}\label{eq1}$$
其中$$\bar{w}_{mi} = e^{-y_if_{m-1}(x_i)}$$.可以看出，$$\bar{w}_{mi}$$只与$$f_{m-1}(x_i)$$有关，与$$\alpha, G$$无关，所以暂时无视这些系数$$\bar{w}_{mi}$$。先求使得\eqref{eq1}式得到最小的$$\alpha, G$$，就是$\alpha_m$和$G_m(x)$。

1. 分两步，首先求$G_m(x)$：对任意$\alpha>0$，使\eqref{eq1}式达到最小的$G(x)$由下式得到：
$$G_m(x) = \arg\min\limits_{G}\sum\limits_{i=1}^N\bar{w}_{mi}I(y_i\neq G(x_i)) \tag{2}\label{eq2}$$
其中$$\bar{w}_{mi} = e^{-y_if_{m-1}(x_i)}$$。比较容易理解，对于$\alpha>0$，系数$\bar{w}>0$，$$\sum\limits_{i=1}^N\bar{w}_{mi}I(y_i\neq G(x_i))$$越小，说明满足$$\sum\limits_{i=1}^N\bar{w}_{mi}e^{\alpha}$$的$i$越少，$\bar{w}$增大（乘以一个大于1的数）的样本数越少。
**所以，$G_m$就是第$m$轮迭代，加权训练数据集，误分类比率最小的分类器。**
2. 然后求$\alpha_m$：
$$\begin{array}{rl}
\sum\limits_{i=1}^N\bar{w}_{mi}e^{-y_i\alpha G_m(x_i)} & = \sum\limits_{y_i=G_m(x_i)}\bar{w}_{mi}e^{-\alpha} + \sum\limits_{y_i\neq G_m(x_i)}\bar{w}_{mi}e^{\alpha} \\
 & = \sum\limits_{i=1}^N\bar{w}_{mi}(\frac{\sum\limits_{y_i=G_m(x_i)}\bar{w}_{mi}}{\sum\limits_{i=1}^N\bar{w}_{mi}}e^{-\alpha} + \frac{\sum\limits_{y_i\neq G_m(x_i)}\bar{w}_{mi}}{\sum\limits_{i=1}^N\bar{w}_{mi}}e^{\alpha}) \tag{3}\label{eq3}
\end{array}$$
这里如果令$$e_m = \frac{\sum\limits_{y_i\neq G_m(x_i)}\bar{w}_{mi}}{\sum\limits_{i=1}^N\bar{w}_{mi}} = \frac{\sum\limits_{i=1}^N\bar{w}_{mi}I(y_i\neq G_m(x_i))}{\sum\limits_{i=1}^N\bar{w}_{mi}} = \sum\limits_{i=1}^nw_{mi}I(y_i\neq G_m(x_i))$$，恰好就是误分类率；对\eqref{eq3}式关于$\alpha$求导并使导数为0，得到
$$\alpha_m = \frac{1}{2}\log\frac{1-e_m}{e_m}$$
因为损失函数\eqref{eq3}的二次导数>0，所以极值点是极小值点。

最后再看权重的更新：由$$f_m(x_i) = f_{m-1}(x_i)+\alpha_m G_m(x_i)$$以及$$\bar{w}_{mi} = e^{-y_if_{m-1}(x_i)}$$可得，
$$\bar{w}_{m+1,i} = \bar{w}_{m,i}e^{-y_i\alpha_m G_m(x_i)}$$

__注意：__ 
- 一般而言$\alpha>0$，因为我们总是找一个弱分类器，即误分类率略小于0.5即可。 
- $w$是样本的权重>0，不是分类器的参数。前向分布算法的每一次迭代，\eqref{eq1}要找使得损失函数最小的分类器参数（也就是确定分类器），这里完全**不是**求样本的权重。
- 每一轮迭代$m$，通过极小化指数损失函数\eqref{eq1}而得到的分类器\eqref{eq2}，是使第$m$轮加权的样本数据误分类之和最小的分类器。
- 在第$m-1$轮就已经确定了样本的权重$$w_{mi}$$，在第$m$轮时$$w_{mi}$$作为常数来考虑，权重只与前一轮的参数有关系，然后在第$m$轮再确定的$G_m$和$\alpha_m$



# Adaboost的优缺点
来自参考资料3：
- 优点：
    - 快, 简单
    - 需要调参的参数很少，除了弱学习器数量$M$
    - Adaboost不容易过拟合
- 缺点：
    - 对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性；


# 参考资料
1. 《统计学习方法》李航，P137-146.
1. [AdaBoost原理详解](https://www.cnblogs.com/ScorpioLu/p/8295990.html)
1. [BOOSTING(Adaboost algorithm)](http://www-math.mit.edu/~rothvoss/18.304.3PM/Presentations/1-Eric-Boosting304FinalRpdf.pdf)