---
title: 变分推断
date: 2020-08-23
updated: 2020-08-23
categories:
    - 机器学习
tags:
    - 近似推断
mathjax: true
---

# 变分推断的由来

从贝叶斯角度思考，inference（推断）指的是求出后验分布$p(\theta|X) = p(X|\theta)p(\theta)/p(X)$，常常参数空间（或隐变量所在的空格）非常复杂、维度非常高，无法求出积分$p(X) = \int_{\theta}p(X|\theta)p(\theta)d\theta$，也就无法求出后验$p(\theta|X)$，无法精确的求出inference，只能做近似的inference。近似inference里面有确定性近似和随机近似两种，这篇介绍的变分推断属于确定性近似推断，随机近似推断指的是MCMC等方法。

# 经典变分推断的推导

用$X$表示观测数据；$Z$表示隐变量+参数，这里于EM算法、GMM模型的符号定义不同，之前$Z$只表示隐变量，这里的$Z$是隐变量以及参数，相当于这里把参数也看作是随机变量了；记$(X, Z)$为完整数据。回顾EM算法：
$$
\begin{array}{rl}
\log p(X) & = \int_zq(Z)\cdot\log\frac{p(X,Z)}{q(Z)}dz - \int_zq(Z)\log\frac{p(Z|X)}{q(Z)}dz\\
& = ELBO + KL(q||p)\\
& = L(q) + KL(q||p)\\
& \ge L(q)
\end{array}
$$
这里用定义$L(q)$来代替ELBO，$L$的输入是概率密度函数$q$，$q(Z)$是一个变分。等式左边的$\log p(X)$跟$q$没有关系，当$X$固定，等式右边也是固定的值。在等式右边，因为KL divergence是非负的，所以无论$q$怎么变，$L(q)$顶多能跟$\log p(X)$一样大。

**因为后验$p(Z|X)$不好求（intractable），我们用$q(Z)$来近似**：找到一个$q(Z)\approx p(Z|X)$. 如果$~q~$跟$~p~$越相似，KL divergence就越接近于0. 所以只需要最大化$L(q)$，就可以找到那个$q(Z)$了。
$$
\hat{q}(Z) = \arg\max\limits_{q(z)} L(q)
$$

经典的变分推断是基于**平均场理论**，假设$q(Z)$可以划分成m个相互独立的组$q(Z) = \prod\limits_{j=1}^m q_j(Z_j)$，**这里的$Z_j$表示的是多个维度的集合**。我们固定m-1个$q_1, \dots, q_{j-1}, q_{j+1}, \dots, q_m$，求$q_j(Z_j)$：

$$
\begin{array}{rl}
L(q) &= \int_Zq(Z)\log p(X,Z)dZ - \int_Zq(Z)\log q(Z)dZ = (1) - (2)\\
(1)& =\int_Z \prod\limits_{i=1}^m q_i(Z_i)\log p(X,Z)dZ_1dZ_2\dots d_{Z_m}\\
& = \int_Zq_j(Z_j)\prod\limits_{i\neq j}q_i(Z_i)\log p(X,Z)dZ_1\dots d{Z_m}\\
& = \int_{Z_j}q_j(Z_j)(\int_{Z_i\neq Z_j}\log p(X,Z)\prod\limits_{i}q_i(Z_i)dZ_i)dZ_j\\
& = \int_{Z_j}q_j(Z_j)\cdot E_{\prod\limits_{i\neq j}q_i(Z_i)}[\log p(X,Z)]dZ_j\\
(2) & = \int_Z\prod\limits_{i=1}^m q_i(Z_i)\cdot \log\prod\limits_{i=1}^m q_i(Z_i)dZ\\
& = \int_Z\prod\limits_{i=1}^m q_i(Z_i)\cdot \sum\limits_{i=1}^m\log q_i(Z_i)dZ\\
& = \int_Z\prod\limits_{i=1}^m q_i(Z_i)\cdot[\log q_1(Z_1)+ \dots + \log q_m(Z_m)]dZ\\
& = \int_Z\prod\limits_{i=1}^m q_i(Z_i)\cdot \log q_1(Z_1)dZ + \dots \\
& = \int_{Z_1Z_2\dots Z_m} q_1q_2\dots q_m\cdot \log q_1dZ_1dZ_2\dots dZ_m + \dots\\
& = \int_{Z_1} q_1\log q_1dZ_1\int_{Z_2}q_2dZ_2\dots\int_{Z_m}q_mdZ_m + \dots\\
& = \int_{Z_1}q_1\log q_1dZ_1 + \dots + \int_{Z_m}q_m\log q_mdZ_m\\
& = \sum\limits_{i=1}^m\int_{Z_i}q_i(Z_i)\log q_i(Z_i)dZ_i\\
& = \int_{Z_j}q_j(Z_j)\log q_j(Z_j)dZ_j + C~(\text{only care}~q_j)
\end{array}
$$

表达式（1）和（2）都写成了$\int_{Z_j}q_j(Z_j){\color{red}{A}}dZ_j$的形式，我们把（1）式的里面也写成log的形式
$$
\begin{array}{rl}
(1) & = \int_{Z_j}q_j(Z_j)\cdot\Bigl( E_{\prod\limits_{i\neq j}q_i(Z_i)}[\log p(X,Z)]\Bigr)dZ_j\\
& := \int_{Z_j}q_j(Z_j)\cdot\Bigl( \log\hat{p}(X,Z_j)\Bigr)dZ_j\\
(1) - (2) & = \int_{Z_j}q_j(Z_j)\log\frac{\hat{p}(X,Z_j)}{q_j(Z_j)}dZ_j~(\text{only care}~q_j)\\
& = -KL(q_j||\hat{p}(X,Z_j))\\
& \le 0
\end{array}
$$
所以当$q_j(Z_j)=\hat{p}(X,Z_j)$时，$L(q)$达到最大。

# 经典变分推断Revisited

经典的变分推断是基于平均场理论的坐标上升法。平均场理论的假设$q(Z) = \prod\limits_{j=1}^m q_j(Z_j)$是比较强的假设：$Z_j$之间是相互独立的（每个$Z_j$不是单一维度，是一部分维度的集合），这样做是为了简化；但是实际上$Z_i$之间往往是有关联的。

EM算法与变分推断，运用的方法是相似的，都是把log-likelihood写成ELBO+KL divergence. 变分推断关注的是后验概率，所以这里就不单独考虑参数。

变分推断的本意是后验概率$p(Z|X)$无法求得，就用概率$q(Z)$去近似，即寻找$\hat{q} = \arg\min\limits_{q} KL(q||p)=\arg\max\limits_q L(q)$，就可以认为这个q接近于p. 但前面由于平均场理论的简化，把q分成了几个$q_j$的乘积，然后逐个的优化$q_j$：
$$
\log q_j(Z_j) = E_{\prod\limits_{i\neq j}q_i(Z_i)}[\log p(X,Z)] + C
$$
这个方法是**迭代的**去找到每个$q_j(Z_j)$，把$q_j(Z_j)$展开：
$$
\log \hat{q}_1(Z_1) = \int_{q_2}\int_{q_3}\dots\int_{q_m}q_1q_2,\dots,q_m \log p(X,Z)dq_1dq_2\dots dq_m\\
\log \hat{q}_2(Z_2) = \int_{\hat{q}_1}\int_{q_3}\dots\int_{q_m}\hat{q}_1q_3,\dots,q_m \log p(X,Z)d\hat{q}_1dq_3\dots dq_m\\
\dots\\
\log \hat{q}_m(Z_m) = \int_{\hat{q}_1}\int_{\hat{q}_2}\dots\int_{\hat{q}_{m-1}}\hat{q}_1\hat{q}_2,\dots,\hat{q}_{m-1} \log p(X,Z)d\hat{q}_1d\hat{q}_3\dots d\hat{q}_m\\
$$
这也是坐标上升法的思想。以上是一轮迭代，每轮迭代之后，可以通过检验$L(q)$是否持续变大，来判断迭代是否中止。

## 经典变分推断的缺点

- 经典变分推断是基于平均场理论，这个假设太强，m份$Z_j$之间是独立的。假设Z非常复杂，比如一个神经网络，每一个维度是一个神经元，那它们相互连接，平均场理论不再适用。
- 即使可以写成$\prod\limits_{j=1}^mq_j(Z_j)$的形式，在坐标上升法求每个$q_j(Z_j)$的时候，要求m-1重积分，这个基本上是intractable的。（前面提到的后验是intractable的，也是因为积分$p(X) = \int_{Z}p(X|Z)p(Z)dZ$难以求得）

# 随机梯度变分推断

待续