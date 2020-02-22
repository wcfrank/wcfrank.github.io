---
title: xgboost
date: 2020-02-19
updated: 2020-02-22
categories:
    - 机器学习
tags:
    - Ensemble Learning
	- Boosting
mathjax: true
---

xgboost与GBDT类似都是一种boosting方法，在之前写GBDT的时候有介绍过一点xgboost的思想，这里认真研究一下。

# 1. xgboost原理

> Adaboost算法是__模型为加法模型的__、__损失函数为指数函数的__、__学习方法为前向分布算法__的二分类问题。-- from本站《Adaboost》博客

xgboost同样也是加法模型、损失函数可以是不同类型的convex函数、学习方法为前向分布算法。继续借用这三点来解释xgboost：

## 1.1 加法模型

xgboost是由T个基模型组成的加法模型，假设第t次迭代要训练的基模型（e.g.决策树）是$f_t(x)$，则
$$
\hat{y}_i^{(t)}=\sum\limits_{k=1}^{t}f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i)
$$
即对于任意样本$i$，第t次迭代后的预测值=前t-1颗树的预测值+第t棵树的预测值

## 1.2 目标函数：正则化&泰勒展开

> [1] 模型的预测精度由模型的偏差和方差共同决定，损失函数代表了模型的偏差，想要方差小则需要在目标函数中添加正则项，用于防止过拟合。 

xgboost的一大改进是在目标函数中加入**正则项**，防止模型过拟合。损失函数可以广义的定义为$\sum\limits_{i=1}^nl(y_i,\hat{y_i})$，那么目标函数可定义为
$$
Obj = \sum\limits_{i=1}^nl(y_i, \hat{y_i}) + \sum_{k=1}^T\Omega(f_k)
$$
另一个改进是损失函数的**泰勒展开**：加法模型$\hat{y}_i^{(t)}= \hat{y}_i^{(t-1)} + f_t(x_i)$，预测值的增量即为$f_t(x_i)$，变化后的预测值对目标函数的影响为
$$
\begin{align}
Obj^{(t)} & = \sum\limits_{i=1}^nl(y_i, \hat{y_i}^{(t)}) + \sum\limits_{k=1}^{t}\Omega(f_k) \\
 & = \sum\limits_{i=1}^nl(y_i, \hat{y_i}^{(t-1)}+f_t(x_i))+ \sum\limits_{k=1}^{t}\Omega(f_k)\\ 
 & = \sum\limits_{i=1}^nl(y_i, \hat{y_i}^{(t-1)}+f_t(x_i))+ \Omega(f_t) + constant
\end{align}
$$
第t步迭代的时候只关心变量$f_t(x_i)$，t-1步之前的都是已知量，所以$\sum\limits_{k=1}^{t-1}\Omega(f_k)$看作常数。损失函数增加了$f_t(x_i)$，使用泰勒展开：
$$
\begin{align}
f(x_0+\Delta x) & = \sum\limits_{i=0}^n\frac{f^{(i)}(x_0)}{i!}\Delta x^i + o(x^n) \\
 & \approx f(x_0) + f'(x_0)\Delta x + \frac{1}{2}f''(x_0)\Delta x^2
\end{align}
$$
得到样本i在第t的迭代之后的预测值：
$$
\begin{align}
l(y_i, \hat{y_i}^{(t)}) & = l(y_i, \hat{y_i}^{(t-1)}+f_t(x_i)) \\
 & \approx l(y_i, \hat{y_i}^{(t-1)}) +  g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)\\
\end{align}
$$
其中$g_i$为损失函数在$\hat{y_i}^{(t-1)}$点的一阶导，$h_i$为损失函数在$\hat{y_i}^{(t-1)}$点的二阶导。总结一下：**泰勒展开是关于损失函数，求导是对上一轮迭代的预测值求一阶、二阶导数。**目标函数可写为
$$
Obj^{(t)}\approx\sum\limits_{i=1}^n[l(y_i,\hat{y_i}^{t-1})+g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)] +\Omega(f_t) + constant
$$
因为第t步时$\hat{y_i}^{(t-1)}$是已知值，所以$l(y_i, \hat{y_i}^{t-1})$可看作常数
$$
Obj^{(t)} \approx \sum\limits_{i=1}^n[g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t) + constant
$$
极小化目标函数$Obj^{(t)}$，就是找到第t步最优的变量$f_t(x)$。最后通过加法模型得到整体模型。

## 1.3 前向分布：学习过程

虽然xgboost基模型不仅支持决策树，还支持线性模型，这里是介绍树模型。在第t步，如何学习到一棵树，以及叶子节点对应的权重。

### 1.3.1 定义一棵树

- 样本与叶子节点的mapping关系q
- 叶子节点的权重w

![sample mapping to leaf nodes](https://pic2.zhimg.com/80/v2-69d198a1de945d849364e11e6b048579_hd.jpg)



# References

1. (重点参考) [深入理解XGBoost](https://zhuanlan.zhihu.com/p/83901304)
2. [前向分步算法：AdaBoost，GBDT和XGBoost算法](https://www.cnblogs.com/LittleHann/p/7512397.html)

