---
title: PCA
date: 2020-04-03
updated: 2020-04-03
categories:
    - 机器学习
tags:
    - 降维
mathjax: true
---

# PCA原理

PCA全称为Principal Component Analysis，主成分分析。什么叫做主成分呢，主成分跟降维什么关系？

PCA的思路就是进行线性变换，对原始特征空间进行重构，在一组线性无关的标准正交基下，挑选一部分最重要的维度。

## 基础知识

假设有n个样本，每个样本的维度为p，则
$$
x=(x_1,x_2,\dots,x_n)^T=
\begin{pmatrix}x_1^T\\x_2^T\\\dots\\x_n^T\end{pmatrix}
=\begin{pmatrix}x_{11}& x_{12}&\dots& x_{1p}\\x_{21}&x_{22}&\dots&x_{2p}\\\dots&\dots&\dots&\dots\\x_{n1}&x_{n2}&\dots&x_{np}\end{pmatrix}
$$
- 样本均值：$\mu=\frac{1}{n}\sum\limits_{i=1}^n x_i$

- 样本协方差矩阵covariance matrix：$S=\frac{1}{n}\sum\limits_{i=1}^n(x_i-\mu)(x_i-\mu)^T$，$x_i$是px1维的，协方差矩阵维pxp维的。



按照[1]和[2]的思路，我们从两个角度来解释PCA：最大投影方差&最小构建误差

## 最大投影方差

为什么是最大化方差？有种说法是信号的方差较大、噪声的方差较小。最大化方差是把有用的"信号"信息提取出来，舍弃没用的“噪声”信息。样本在某个基方向上的坐标为投影，投影的方差就是在某个基方向上坐标的方差。投影方差达到最大的前k个基方向，为前k个主成分。

>  [5]降维问题的优化目标：将一组p维向量降为k维，其目标是选择k个单位正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大。

假设某一个基方向为$w$，样本在这个基方向上的投影为$w^Tx$（即在$w$方向上的坐标）。 样本在这个方向投影的方差可表示为
$$
Var(w^Tx) = \frac{1}{n}\sum\limits_{i=1}^n(w^Tx_i-\mu)^2
$$
另外，PCA有一个必须的步骤是中心化，即样本在原坐标下减去均值$\mu=\frac{1}{n}\sum\limits_{i} x_i$。所以样本在线性变换之后的方差可表示为
$$
Var(w^Tx) = \frac{1}{n}\sum\limits_{i=1}^n(w^Tx_i)^2=w^T(\frac{1}{n}\sum\limits_{i=1}^n x_ix_i^T)w=w^TSw
$$

最大投影方差变成了寻找基向量$w$（且单位长度$w^Tw=1$），使得$w=\arg\max w^TSw$. 转化为一个优化问题：
$$
\begin{array}{rl}
\max\limits_{w} & w^TSw \\
s.t. & ||w||=1
\end{array}
$$
使用拉格朗日乘子法将其变为无约束优化问题：
$$
L(w,\lambda) = w^TSw -\lambda(1-w^Tw)\\
\frac{\partial L}{\partial w} = 2Sw - 2\lambda w=0\\
Sw=\lambda w
$$
可以发现，求极值等价于求**协方差矩阵S**的特征值和特征向量。如果基向量$w$为极值，同时也是S的特征向量，投影到该基向量的方差取得极大值，值为$w^TSw=w^T\lambda w=\lambda$. 我们要找的最大投影方差，基向量就是样本协方差矩阵的特征向量（i.e.一个主成分），投影方差的值就是该特征向量对应的特征值。

> 实对称矩阵的不同特征值对应的特征向量互相正交。n阶实对称矩阵A必可对角化，且对角阵上的元素即为矩阵本身特征值。n*n的实对称矩阵一定存在 n个相互正交的特征向量

将特征值按照从大到小排列，选择前k个特征值对应的特征向量（正交化），这就是k个主成分，用这k个主成分来表示样本，达到降维的目的（$p\rightarrow k$）。[3]第一个基方向是原始数据中方差最大的方向，第二个基方向的选取是与第一个基方向正交的平面中方差最大的，第三个轴是与第1,2个轴正交的平面中方差最大的...

### 特征值重数与主成分

关于S的特征值的重数可能大于1，即同一个特征值会对应多个线性无关的特征向量。PCA选择前k个特征向量的时候，是包括特征值的重数还是选择k个不同取值的特征值？

PCA在选择主成分的时候，对选择对应特征值最大的k个特征向量，有的特征向量对应相同的特征值，这些特征向量可以都被选择。但是，实对称矩阵相同特征值对应的特征向量是线性无关，但未必正交。所以需要对相同特征值的特征向量做正交化，才可以作为基方向被PCA选择。这里不多做展开，可以参考[3, 4]：

> [4] 任意的 N×N 实对称矩阵都有 N 个线性无关的特征向量。并且这些特征向量都可以正交单位化而得到一组正交且模为 1 的向量。故实对称矩阵 A 可被分解成$\mathbf{A}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{T}$，其中其中 Q为正交矩阵，Λ 为实对角矩阵。

[3]把实对称矩阵A做特征值分解，Q就是A的（正交的）特征向量所组成的矩阵，Λ对角线上的元素就是A的特征值。A经过Q变成对角阵，也呼应了前文引用[5]中优化目标：字段内的方差最大化，字段间的协方差为0. 

### 为什么要中心化和scaling

中心化是每个特征减去这个特征的均值。[6]协方差矩阵的计算，本身就蕴含着中心化的思想$S=\frac{1}{n}\sum\limits_{i=1}^n(x_i-\mu)(x_i-\mu)^T$，分母是n-1还是n不重要。中心化对协方差矩阵没有影响。

Scaling同样也是推荐的preprocessing步骤，因为feature的取值范围可能相差很大，这会影响有些基方向的方差大小，如果没有做标准化，PCA算出的向量和长轴会有偏差。[PCA降维之前为什么要先标准化？](http://sofasofa.io/forum_main_post.php?postid=1000375)



**Note：**寻找的基向量为单位长度，这一点也很重要，这样才有拉格朗日乘子法的构造。

## 最小构建误差

# PCA算法步骤

输入：数据集$x=\{x_1,x_2,\dots,x_n\}$ ，$x_i$的维度为p维，需要降到k维。

1) 去平均值(即去中心化)，即每个**特征**减去各自的平均值。

2) 计算协方差矩阵$S=\frac{1}{n}xx^T$,注：这里除或不除样本数量n或n-1,其实对求出的特征向量没有影响。

3) 用特征值分解方法求协方差矩阵$S=\frac{1}{n}xx^T$的特征值与特征向量。

4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为行向量组成特征向量矩阵P。

5) 将数据转换到k个特征向量构建的新空间中，即Y=Px。





、是否需要scaling

为什么要协方差

如果基向量不是单位长度，会怎么样？

# 使用&优劣

# References

1. [机器学习-白板推导系列(五)-降维（Dimensionality Reduction）](https://www.bilibili.com/video/BV1vW411S7tH?from=search&seid=15511856047644180318)
2. [主成分分析PCA算法：为什么去均值以后的高维矩阵乘以其协方差矩阵的特征向量矩阵就是“投影”？](https://www.zhihu.com/question/30094611/answer/275172932)
3. [主成分分析（PCA）原理详解](https://zhuanlan.zhihu.com/p/37777074)
4. [特征分解](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3)
5. [PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)
6. [How does centering make a difference in PCA (for SVD and eigen decomposition)?](https://stats.stackexchange.com/a/189902)
7. [主成分分析PCA算法：为什么要对数据矩阵进行均值化](https://www.zhihu.com/question/40956812)
8. [主成分分析（Principal components analysis）-最大方差解释](https://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html)

