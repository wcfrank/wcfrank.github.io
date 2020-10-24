---
title: 异常检测之PCA
date: 2020-10-22
updated: 2020-10-22
categories:
    - 异常检测
tags:
    - PCA
    - 线性模型
mathjax: true
---

PCA方法适用于**线性模型**。一般的，PCA的主要作用是降维（在之前的文章中已经讨论过），但也可以用于异常检测，本文就来解释一下PCA如何用来进行异常检测，本文复述了参考资料[2]中的Section3.3

# PCA回顾

PCA可以找到任意维度的最优的超平面，来表示原始数据$X_{n\times d}=[x_1,x_2,\dots,x_n]^T$. 也就是说，PCA可以找到$k$ -维超平面($k<d$)，使得剩下的$(d-k)$-维投影到这k维上去的投影误差最小。

协方差矩阵是$d\times d$维的，其中第$(i,j)$个元素表示对于这n个观测样本，第$i$维和第$j$维特征的协方差。协方差矩阵记为$S=\frac{1}{n}X^TX$. 

协方差矩阵是对称的、半正定的，所以可以特征值分解为$S=P\Lambda P^T$，其中$\Lambda$为对角阵，对角线上的元素为$S$的特征值，矩阵$P_{d\times d}$的列是对应特征值的特征向量。可以选择k个最大特征值对应的特征向量（正交的），构成k维子空间，来近似表示原始数据。所以PCA首先需要根据协方差矩阵$S$的一部分特征向量为正交基向量，构造新的子空间；特征向量是子空间的基，是原始数据$x_i$被投影的方向。

PCA有以下两个重要性质[2]:

> 1. If the data is transformed to the axis-system corresponding to the orthogonal eigen- vectors, the variance of the transformed data along each axis (eigenvector) is equal to the corresponding eigenvalue. The covariances of the transformed data in this new representation are 0.
>
> 2. Since the variances of the transformed data along the eigenvectors with small eigen- values are low, significant deviations of the transformed data from the mean values along these directions may represent outliers.

假设原始数据$x_i$，是1xd维的，在完整的特征向量构成的空间下的坐标为$x'_i=x_iP$，还是1xd维的。同理$X'=XP$. 对于新的表示$X'$，不同特征之间的协方差为0，同一特征的方差为特征值（呼应前面性质2）。

# PCA用于异常检测

由性质1，经过空间变换之后，每个特征的方差为对应的特征值。假设第$j$个特征值非常小，则变换之后$x'_{ij}$在固定的$j$以及不同的$i$时，取值的变化不会很大。因为PCA的数据会去中心化，即每个特征减去各自的均值，变换之后的$x'_{ij}$应该在均值0的附近，否则就是**异常点(outlier)**。

实际情况往往会有一大部分的特征值非常的小，这意味着大多数的数据可以在更低维度的子空间下被表示。而那些特征值非常小的维度，因为数据的变化太小，可以被舍弃。从异常检测的角度上看，这是件好事：如果有的观测样本无法在更低维度的子空间下被表示，说明在其他某个特征值很小的方向上变化较大，这个就是异常点。

再具体一点：假设大多数的variance在$k$维子空间被捕捉到，剩下的$(d-k)$维空间上几乎没有variance。我们计算观测数据到这个$k$维超平面的投影距离（超平面经过样本均值点），如果观测数据到超平面的距离过大（说明在其他维度空间上还有deviation），说明这个观测数据为异常点。那么如何度量观测数据到超平面的距离？

## Hard PCA

观测数据到超平面的squared欧式距离，可以分解为$(d-k)$个最小的特征值对应的特征向量方向的距离平方之和；同时每个方向的距离要除以对应的特征值，为了标准化variance。假设特征值是按照从大到小排序的，$\lambda_j$为第j个特征值，$e_j$为第j个特征向量：
$$
Score({x})= \sum\limits_{j=d-k+1}^d\frac{|({x}-\mu)\cdot e_j|^2}{\lambda_j}
$$
所以这个Score是沿着$(d-k)$个**没有**被选做主成分的特征向量方向来计算的，计算的结果是到$k$个主成分构成的子空间的距离。每一项做scaling的原因是：在特征向量方向上的variance就是特征值，因此一个比较小的特征值如果有了一个大的deviation，会对Score贡献更多。

## Soft PCA & Mahalanobis距离

前面的方法是直接选择前$k$大的特征值对应的特征向量，做为子空间的基，然后计算剩下$(d-k)$个方向的投影距离的加权之和。

另外一种方法是选择所有特征向量的方向，计算沿着每个方向从观测数据点到centroid点$\mu$的标准化的距离（因为观测数据X是经过去中心化的，所以centroid点$\mu$是原点）：
$$
Score(x) = \sum\limits_{j=1}^d\frac{|(x-\mu)\cdot e_j|^2}{\lambda_j}
$$
分子是特征向量方向的投影长度，就是在新的空间下的坐标；分母是标准差，如果把数据的每一列提前标准化之后，分母就是1了。这样Score就变成了新坐标的平方和，即为欧式距离。再捋一遍求解过程：

1. 计算原始观测数据X的协方差矩阵S，并将S对角化$S=P\Lambda P^T$，
2. 将X坐标变换到新的空间下$X'=XP$，
3. 对$X'$的每一列都除以它对应的标准差（特征值），将每一列标准化，得到新的$X'$，
4. 对于新$X'$的每一行（每个样本），求解它到centroid点的欧式距离的平方。

Score的取值主要由$\lambda_j$比较小、而且deviation比较大的样本造成。所以第3步的标准化将每一列拉到了同一个scale来计算，这个方法是给予每一列一个soft的权重，而不是直接hard的选择前k列。这个Score是Mahalanobis距离。

## Hard v.s. Soft

Soft PCA是对每一个特征向量的方向一个权重，而Hard PCA是预先选择出一部分特征向量。如果某个观测数据是在特征值最大的特征向量上面有很大的deviation，按照Hard PCA会不考虑这个特征向量的方向，这个异常点就会被忽略。所以Soft比Hard PCA更好。

Hard PCA（不考虑那个Score的计算方法）考虑的是reconstruction error，使得样本在低维空间被表示；但同时Hard PCA也引入了额外的参数k。

## Sensitivity to Noise

PCA一般而言对异常点比较稳定，不太容易受到异常点的影响。这是因为PCA计算的是最优的超平面，而不是某个特定的变量。当观测数据加入了一些噪声数据之后，最优超平面不会产生非常剧烈的变化。不过有时候噪声数据确实会造成一些数据问题，我们依然有办法解决：

- 首先使用PCA异常检测的方法发现明显的异常点
- 删掉这些异常点，重新求协方差矩阵等等，这样数据会更加的robust

- 使用更新后的协方差矩阵，重新计算异常点Score

以上这个过程可以**迭代**的求解，在每一次迭代中，去掉明显的异常点，构建更robust的PCA。

## Normalization Issues

PCA一般只是去中心化，但如果特征之间的scale差距很大，也可能使结果变得很差。假设有两个特征Age和Salary，Salary的scale往往比Age大的多，PCA更容易得到一个跟Salary方向平行的主成分，却没有考虑到Age跟Salary之间有很高的相关性，这不利于异常检测。可以通过归一化每个特征，即每个特征除以它的标准差，使得每个特征维度上的variance都是单位化的。这样我们就不必使用协方差矩阵，也可以使用correlation matrix了。（$Corr(X,Y) = \frac{Cov(X,Y)}{\sigma_x \sigma_y}$）

## Regularization Issues

当观测数据的样本数量很少时，协方差矩阵不能很好的反应出真实情况。极端情况下，样本数量太少，使得有些维度的variance为0，低估了真实的取值变化。

一种方法是**加入正则项避免过拟合**，这里的正则项类似于Laplacian smoothing，调整协方差矩阵为$(S+\alpha I)$，其中$I$是一个dxd维的单位矩阵，$\alpha>0$. 使用调整后的协方差矩阵$(S+\alpha I)$计算异常Score。这样做的原因是，在计算Score之前，在每个维度上加入一些variance为$\alpha$的噪声。

另一种方法是使用cross-validation，数据被分成m份，PCA通过其中的m-1份构造子空间，然后用剩下的一份数据来求Score。

## How many Eigenvectors?

虽然在用PCA做异常检测的时候，可以使用soft的方法对每一个特征向量做加权求和得到异常Score。但有时候也需要提前指定一定数量的特征向量，作为主成分。那么选择多少个特征向量比较合适呢？

一个事实是，真实数据的大多数特征值都比较小，数据大多数的变化都来自于少数几个特征向量的方向。

## 非线性数据

Kernel PCA

# References

1. [数据挖掘中常见的「异常检测」算法有哪些？](https://www.zhihu.com/question/280696035/answer/417091151)
2. [Outlier Analysis. Section 3.3](https://www.academia.edu/download/57870486/Outlier-Analysis.pdf)

