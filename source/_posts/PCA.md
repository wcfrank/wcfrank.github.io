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

PCA的思路就是进行线性变换，在一组标准正交基下，挑选一部分最重要的维度。

## 基础知识

假设有n个样本，每个样本的维度为p，则
$$
x=(x_1,x_2,\dots,x_n)^T=
\begin{pmatrix}x_1^T\\x_2^T\\\dots\\x_n^T\end{pmatrix}
=\begin{pmatrix}x_{11}& x_{12}&\dots& x_{1p}\\x_{21}&x_{22}&\dots&x_{2p}\\\dots&\dots&\dots&\dots\\x_{n1}&x_{n2}&\dots&x_{np}\end{pmatrix}
$$
- 样本均值：$\mu=\frac{1}{n}\sum\limits_{i=1}^n x_i$

- 样本协方差矩阵covariance matrix：$S=\frac{1}{n}\sum\limits_{i=1}^n(x_i-\mu)(x_i-\mu)^T$，$x_i$是px1维的，协方差矩阵维pxp维的。



按照[1]和[2]的思路，我们从两个角度来解释PCA

## 最大投影方差

为什么是最大化方差？有种说法是信号的方差较大、噪声的方差较小。最大化方差是把有用的"信号"信息提取出来，舍弃没用的“噪声”信息。样本在某个基方向上的坐标为投影，投影的方差就是在某个基方向上坐标的方差。投影方差达到最大的前k个基方向，为前k个主成分。

假设某一个基方向为$w$，样本在这个基方向上的投影为$w^Tx$（即在$w$方向上的坐标）。 样本在这个方向投影的方差可表示为
$$
Var(w^Tx) = \frac{1}{n}\sum\limits_{i=1}^n(w^Tx_i-\mu)^2
$$
另外，PCA有一个必须的步骤是中心化，即样本在原坐标下减去均值$\mu=\frac{1}{n}\sum\limits_{i} x_i$。所以样本在线性变换之后的方差可表示为
$$
Var(w^Tx) = \frac{1}{n}\sum\limits_{i=1}^n(w^Tx_i)^2=w^T(\frac{1}{n}\sum\limits_{i=1}^n x_ix_i^T)w=w^TSw
$$


## 最小构建误差

# PCA算法步骤

# 为什么要中心化

# 为什么要协方差

# 使用&优劣

# References

1. [机器学习-白板推导系列(五)-降维（Dimensionality Reduction）](https://www.bilibili.com/video/BV1vW411S7tH?from=search&seid=15511856047644180318)
2. [主成分分析PCA算法：为什么去均值以后的高维矩阵乘以其协方差矩阵的特征向量矩阵就是“投影”？](https://www.zhihu.com/question/30094611/answer/275172932)
3. [主成分分析PCA算法：为什么要对数据矩阵进行均值化](https://www.zhihu.com/question/40956812)
4. [主成分分析（Principal components analysis）-最大方差解释](https://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html)

