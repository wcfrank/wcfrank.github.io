---
title: PCA
date: 2020-04-03
updated: 2020-04-10
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

为什么是最大化方差？有种说法是信号的方差较大、噪声的方差较小。最大化方差是把有用的“信号”信息提取出来，舍弃没用的“噪声”信息。样本在某个基方向上的坐标为投影，投影的方差就是在某个基方向上坐标的方差。投影方差达到最大的前k个基方向，为前k个主成分。

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

[3]把实对称矩阵A做特征值分解，Q就是A的（正交的）特征向量所组成的矩阵，Λ对角线上的元素就是A的特征值。A经过Q变成对角阵，也呼应了前文引用[5]中优化目标：**字段内的方差最大化，字段间的协方差为0. **

### 为什么要中心化和scaling

中心化是每个特征减去这个特征的均值。[6-1]协方差矩阵的计算，本身就蕴含着中心化的思想$S=\frac{1}{n}\sum\limits_{i=1}^n(x_i-\mu)(x_i-\mu)^T$，分母是n-1还是n不重要。中心化对协方差矩阵没有影响。不过：

> [6-2] principal components inevitably come through the origin. If you forget to center your data, the 1st principal component may pierce the cloud not along the main direction of the cloud, and will be (for statistics purposes) misleading.
>
> ![before and after centering](https://i.stack.imgur.com/9vENg.jpg)

如果使用特征值分解的求解方法，不提前做中心化，会得到不同的主成分，因为主成分必然经过远点，这会产生误差。

（以下是自己的理解）最大化投影误差$\frac{1}{n}w^Tx^Txw$，如果没有中心化是$\frac{1}{n}w^T(x-\mu)^T(x-\mu)w$，虽然$x$和$x-\mu$是相同的，看似得到的w是一样的，但是这个$w$是在不同的坐标下，一个是中心化之后，一个是之前，这导致了得到的特征向量不同。

> [6-3] 从线性变换的本质来说，PCA就是在线性空间做一个旋转（数据矩阵右乘协方差矩阵的特征向量矩阵），然后取低维子空间（实际上就是前n_components个特征向量张成的子空间）上的投影点来代替原本的点，以达到降维的目的，注意我说的，只做了旋转，没有平移，所以首先你要保证原本空间里的点是以原点为中心分布的，这就是zero mean的目的。

Scaling同样也是推荐的preprocessing步骤，因为feature的取值范围可能相差很大，这会影响有些基方向的方差大小，如果没有做标准化，PCA算出的向量和长轴会有偏差。[PCA降维之前为什么要先标准化？](http://sofasofa.io/forum_main_post.php?postid=1000375)

### 为什么要用协方差

[7] 协方差矩阵已经中心化过；协方差矩阵$\frac{1}{n}xx^T$是对称矩阵，可以转化为协方差矩阵的特征向量求解，而且主成分就是协方差矩阵特征值最大的那几个特征向量。

### 为什么基向量是单位长度

内积只有在某个向量为单位向量的时候可称作是投影，一般性的$a^Tb=|a||b|cos\theta$为a的投影*b的长度。PCA是最大投影方差，所以要规定基向量为单位向量。

PCA算法是其中一个步骤是对协方差进行特征值分解$\mathbf{A}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{T}$，矩阵Q就是一个正交向量构成的矩阵。可将矩阵Q单位化，就得到对应的基向量长度为1了。

## 最小重建cost

![reconstruction error](/Users/Chao/Machine_Learning/Git_Repository/wcfrank.github.io/source/_posts/PCA/reconstruction_error.png)

以上是一个2维的例子，左图为原始的5个样本点，直线方向是第一个主成分；如果降维只用这个主成分表示样本，从2维降至1维，即为右图，5个样本点分布在这条直线上。降维前后，样本的信息必然有所损失，PCA用欧式距离来衡量这个损失。

通过对协方差矩阵进行特征值分解，得到了p组特征值和特征向量$u_1, u_2, \dots, u_p$，选择前k大的特征向量（即k个正交的单位向量），从p维降到k维。在全部p个单位正交向量下，可以“无损失”的重构样本$x_i = \sum\limits_{j=1}^p\alpha_{ij}u_{j}=\sum\limits_{j=1}^p(x_i^Tu_j)u_j$. 如果降维的重构样本$\hat{x}_i = \sum\limits_{j=1}^k\alpha_{ij}u_{j}=\sum\limits_{j=1}^k(x_i^Tu_j)u_j$，原始的样本跟重构之后的样本的差距为
$$
\begin{array}{rl}
Cost & = \frac{1}{n}\sum\limits_{i=1}^n||x_i-\hat{x}_i||^2\\
 & =\frac{1}{n}\sum\limits_{i=1}^n ||\sum\limits_{j=k+1}^p(x_i^Tu_j)u_j||^2\\
 & =\frac{1}{n}\sum\limits_{i=1}^n\sum\limits_{j=k+1}^pu_j^Tx_ix_i^Tu_j\\
 & =\sum\limits_{j=k+1}^pu_j^T(\frac{1}{n}\sum\limits_{i=1}^nx_ix_i^T)u_j\\
 & =\sum\limits_{j=k+1}^pu_j^TSu_j\\
\end{array}
$$
同样可以转化为一个极小化的优化问题
$$
\begin{array}{rl}
\min & \sum\limits_{j=k+1}^pu_j^TSu_j\\
s.t. & u_j^Tu_j=1
\end{array}
$$
因为单位正交向量线性无关，所以向量可以逐个进行优化。有最大投影方差的推导可知，$\sum\limits_{j=k+1}^pu^T_jSu_j=\sum\limits_{j=k+1}^p\lambda_j$，忽略了最小特征值对应的特征向量所在的维度，重建样本与原始样本的差距为$\sum\limits_{j=1k+1}^p\lambda$，最小重建cost目的是最小化这个差距。

**总结：**最大投影方差是找主成分， 最小重建cost是维度缩减。这两个角度本质上是一样的，最小重建cost与降维的思想更加贴近。 

# PCA算法步骤

输入：数据集$x=\{x_1,x_2,\dots,x_n\}$ ，$x_i$的维度为p维，需要降到k维。

1) 去平均值(即去中心化)，即每个**特征**减去各自的平均值。

2) 计算协方差矩阵$S=\frac{1}{n}xx^T$,注：这里除或不除样本数量n或n-1,其实对求出的特征向量没有影响。

3) 用特征值分解方法求协方差矩阵$S=\frac{1}{n}xx^T$的特征值与特征向量。

4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为行向量组成特征向量矩阵P。

5) 将数据转换到k个特征向量构建的新空间中，即Y=Px。

# PCA主成分维度怎么确定

使用PCA从p维降至k维，那k取多少合适呢？

关于最优的k选取，[8-1]的总结比较全面：

> 1. 如果是为了数据可视化，可以降到1维（线），2维（平面），或者3维（立体）。
>
> 2. 如果是为了建立预测模型而降维，比较常用的方法是看多少个主成分解释了多少百分比的方差，常用的比如说99%,95%,90%。
> 3. 另一个方法是Kaiser's Rule，保留所有奇异值大于1的
> 4. 还有个类似elbow method的方法，画出主成分的个数和解释方差百分比的曲线，找出手肘的那个点。

关于上面第2点，可以参考[8-2]中的一个引用cs229的回答：选择k使得（重建cost / total variation）达到最小或者达到一个阈值。这最终可转化为协方差矩阵对角化之后，对角线上元素的计算。

假设阈值设为10%，要找到最小的k，使得
$$
\frac{\frac{1}{n}\sum\limits_{i=1}^n||\sum\limits_{j=1}^p(x_i^Tu_j)u_j-\sum\limits_{j=1}^k(x_i^Tu_j)u_j||^2}{\frac{1}{n}\sum\limits_{i=1}^n||\sum\limits_{j=1}^p(x_i^Tu_j)u_j||^2}\le 10\%
$$

即
$$
1 - \frac{\sum\limits_{i=1}^k\lambda_i}{\sum\limits_{i=1}^p\lambda_i}\le 10\%
$$

# 优劣

优点：

- 不受样本label的限制，以方差衡量信息，
- 各个主成分之间正交，消除原始数据特征的相关性，

- 计算简单，主要运算是特征值分解

局限：

- 基于线性变换的降维方法，

- 主成分分析基于Euclidean distance, 


# References

1. [机器学习-白板推导系列(五)-降维（Dimensionality Reduction）](https://www.bilibili.com/video/BV1vW411S7tH?from=search&seid=15511856047644180318)

2. [主成分分析PCA算法：为什么去均值以后的高维矩阵乘以其协方差矩阵的特征向量矩阵就是“投影”？](https://www.zhihu.com/question/30094611/answer/275172932)

3. [主成分分析（PCA）原理详解](https://zhuanlan.zhihu.com/p/37777074)

4. [特征分解](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3)

5. [PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)

6. [How does centering make a difference in PCA (for SVD and eigen decomposition)?](https://stats.stackexchange.com/a/189902)

   [How does centering the data get rid of the intercept in regression and PCA?](https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca)

   [主成分分析PCA算法：为什么要对数据矩阵进行均值化](https://www.zhihu.com/question/40956812)

7. [求pca的时候，能不能不用协方差矩阵？](https://www.zhihu.com/question/39234760/answer/80323126)

8. [利用PCA降维，降到几维比较合适？](http://sofasofa.io/forum_main_post.php?postid=1000776)

   [主成分分析（PCA）主成分维度怎么选择？](https://www.zhihu.com/question/21980732)

