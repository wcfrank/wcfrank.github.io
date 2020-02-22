---
title: xgboost
date: 2020-02-19
updated: 2020-02-23
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

xgboost是由m个基模型组成的加法模型，假设第t次迭代要训练的基模型（e.g.决策树）是$f_t(x)$，则
$$
\hat{y}_i^{(t)}=\sum\limits_{k=1}^{t}f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i)
$$
即对于任意样本$i$，第t次迭代后的预测值=前t-1颗树的预测值+第t棵树的预测值

## 1.2 目标函数：正则化&泰勒展开

> [1] 模型的预测精度由模型的偏差和方差共同决定，损失函数代表了模型的偏差，想要方差小则需要在目标函数中添加正则项，用于防止过拟合。 

xgboost的一大改进是在目标函数中加入**正则项**，防止模型过拟合。损失函数可以广义的定义为$\sum\limits_{i=1}^nl(y_i,\hat{y_i})$，那么目标函数可定义为
$$
Obj = \sum\limits_{i=1}^nl(y_i, \hat{y_i}) + \sum_{k=1}^m\Omega(f_k)
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
constant项在下文可忽略，因为极小化目标函数$Obj^{(t)}$，就是找到第t步最优的变量$f_t(x_i)$（训练出一棵树$f_t$）。最后通过加法模型得到整体模型。

## 1.3 前向分布：学习过程

虽然xgboost基模型不仅支持决策树，还支持线性模型，这里是介绍树模型。

在第t步，如何学习到一棵树$f_t$：树的结构、叶子节点对应的权重。

### 定义一棵树

- 样本与叶子节点的mapping关系q
- 叶子节点的权重w

![sample mapping to leaf nodes](https://pic2.zhimg.com/80/v2-69d198a1de945d849364e11e6b048579_hd.jpg)



### 定义树之后的目标函数

- 叶子节点归组：一棵树$f_t$确定之后，样本对应的叶子节点就确定了。若干条样本，会在有限个叶子节点中取值，$f_t(x_i) = w_{q(x_i)}$. 依照上图定义一棵树总共有T个叶子节点，令所有属于叶子节点$j$的样本$x_i$集合为$I_j=\{i | q(x_i)=j\}$，所有属于$I_j$的样本取值均为$w_j$.

- 树的复杂度：复杂度由叶子数T和权重w组成，希望树不要有过多的叶子节点，并且节点的不具有过高的权重。
  $$
  \Omega(f_t) = \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw_j^2
  $$

结合以上2点变化，把样本点按照叶子节点归组，目标函数可以写成
$$
\begin{align}
Obj^{(t)} &\approx \sum\limits_{i=1}^n[g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)\\
& = \sum\limits_{i=1}^n[g_iw_{q(x_i)}+\frac{1}{2}h_iw^2_{q(x_i)}] + \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw_j^2\\
& = \sum\limits_{j=1}^T[(\sum\limits_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum\limits_{i\in I_j}h_i+\lambda)w_j^2] + \gamma T\\
& := \sum\limits_{j=1}^T[({\color{red}G_j}w_j+\frac{1}{2}({\color{red}H_j}+\lambda)w_j^2] + \gamma T
\end{align}
$$

- $G_j=\sum\limits_{i\in I_j}g_i$是叶子节点j包含的所有样本的一阶导数之和，一阶导数是$\left.\frac{\partial l(y_i,y)}{\partial y}\right|_{\hat{y_i}^{(t-1)}}$，是t-1步得到的结果，
- $H_j=\sum\limits_{i\in I_j}h_i$是叶子节点j包含的所有样本的二阶导数之和，一阶导数是$\left.\frac{\partial^2 l(y_i,y)}{\partial y^2}\right|_{\hat{y_i}^{(t-1)}}$，也是t-1步得到的结果。

这样目标函数的表达式，只有w是未知变量，其余都是t-1步后已知，是关于w的一元二次表达式。**[3]只要目标函数是凸函数，即二阶导数$H_j>0$，可以保证一元二次表达式的二次项>0，目标函数取到最小值。**

对每个叶子节点j，与j有关的目标函数的部分为$G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2$. 因为叶子节点相互独立，每个叶子节点对应的目标函数达到最小值，整个目标函数$Obj^{(t)}$就达到最小值。可求得节点j的最优权重$w^*_j=-\frac{G_j}{H_j+\lambda}$，树$f_t$的目标函数值为
$$
Obj=-\frac{1}{2}\sum\limits_{j=1}^T\frac{G_j^2}{H_j+\lambda} + \gamma T
$$

### 如何得到最优的树

在确定一颗树之后，可以求得这棵树的最优权重值以及目标函数值，如何确定最优的树呢？一棵树从一个节点（树深为0）开始，逐渐划分节点，生长成新的树。关键就是在每次划分节点的时候，找到最后的划分方式。





## Questions

Hessian？

xgboost梯度的维度？

# References

1. (重点参考) [深入理解XGBoost](https://zhuanlan.zhihu.com/p/83901304)
2. [前向分步算法：AdaBoost，GBDT和XGBoost算法](https://www.cnblogs.com/LittleHann/p/7512397.html)
3. [Custom Loss Functions #353](https://github.com/dmlc/xgboost/issues/353)

