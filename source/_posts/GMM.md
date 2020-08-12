---
title: GMM
date: 2020-08-12
updated: 2020-08-12
categories:
    - 机器学习
tags:
    - EM
    - 混合模型
mathjax: true
---
# GMM介绍

样本的分布有时候比较复杂，无法用单个分布来表示。

- 从几何角度：GMM是多个高斯分布的加权平均，可能由多个不同的分布叠加而成。$p(X) = \sum\limits_{i=k}^K\alpha_kN(\mu_k, \Sigma_k)$, 其中 $\sum\limits_{k=1}^K\alpha_k=1$. 

- 从混合模型角度：除了观测变量$X$；引入隐变量$Z$，$Z$代表对应的样本$X$是属于哪一个高斯分布，GMM的$Z$是离散的随机变量，概率分布为（以下$\sum\limits_{k=1}^Kp_k=1$）

| Z    | C_1  | C_2  | ...  | C_K  |
| ---- | ---- | ---- | ---- | ---- |
| p(z) | p_1  | p_2  | ...  | p_K  |

  如果GMM由K个高斯分布构成，每个样本x既可能属于分布$C_1$, ..., 也可能属于$C_K$，只不过属于不同高斯分布的概率不同。

- 从生成模型角度：对每个样本$x_i$的生成过程如下
  1. 首先按照概率$p_k$选择隐变量Z的分类，假设选择的结果是第h类$Z_i=C_h$，
  2. 然后用第h类对应的高斯分布$N(\mu_h,\Sigma_h)$采样得到样本$x_i$

x的概率密度函数：
$$
p(x) = \sum\limits_{Z}p(x,z) = \sum\limits_{k=1}^Kp(x,z=C_k) = \sum\limits_{k=1}^Kp(z=C_k)p(x|z=C_k)=\sum\limits_{k=1}^Kp_kN(x|\mu_k, \Sigma_k)
$$  
可以发现混合模型角度与几何角度是一致的。

# GMM的求解

## 极大似然无法得到解析解

观测数据$X=\{x_1, x_2,\dots, x_n\}$，完整数据，$(X,Z) = \{(x_1,z_1), (x_2, z_2), \dots, (x_n, z_n)\}$参数$\theta=\{p_1,\dots,p_K,\mu_1,\dots,\mu_K, \Sigma_1,\dots,\Sigma_K\}$，如果用极大似然估计直接求解参数
$$
\hat{\theta}_{MLE} = \arg\max\limits_{\theta}\log P(X) = \arg\max\limits_{\theta}\log\prod\limits_{i=1}^N P(x_i) = \arg\max\limits_{\theta}\sum\limits_{i=1}^N\log P(x_i)
$$
但是，极大似然估计是**无法**得到我们想要的参数结果（因为有隐变量的存在）：
$$
\begin{array}{rl}
\hat{\theta}_{MLE} & = \arg\max\limits_{\theta}\sum\limits_{i=1}^N\log P(x_i) \\ 
 & = \arg\max\limits_{\theta}\sum\limits_{i=1}^N\log\sum\limits_{k=1}^Kp_kN(x_i|\mu_k,\Sigma_K) \\
\end{array}
$$
log函数的后面是一个连加的格式（GMM的隐变量导致的），所以对$\hat{\theta}_{MLE}$求偏导，无法得到解析解。况且，高维的高斯分布本身就很负杂，再乘以一个参数$p_k$，所以更加难以求解。

## 用EM算法求解

含有隐变量Z的混合模型，用EM求解是一种流行且优美的方法。本文用EM算法求解GMM的learning问题。回顾EM算法：
$$
\theta^{(t+1)} = \arg\max E_{z|x,\theta^{(t)}}[\log p(x,z|\theta)] = \arg\max Q(\theta, \theta^{(t)})
$$

 ### E-step


$$
\begin{array}{rl}
Q(\theta, \theta^{(t)}) & = \int_Z\log p(X,Z|\theta)\cdot p(Z|X, \theta^{(t)})dZ \\
 & = \sum\limits_{Z}\log\prod\limits_{i=1}^n p(x_i,z_i|\theta)\cdot \prod\limits_{i=1}^n p(z_i|x_i,\theta^{(t)}) \\
 & = \sum\limits_{z_1, z_2,\dots,z_n}\left(\sum\limits_{i=1}^n\log p(x_i,z_i|\theta)\right)\cdot\prod\limits_{i=1}^n p(z_i|x_i,\theta^{(t)}) \\
\end{array}
$$

其中$\sum\limits_{i=1}^n\log p(x_i,z_i|\theta)$展开会有n项，每一项均与$\prod\limits_{i=1}^n p(z_i|x_i,\theta^{(t)})$相乘（这里以第一项为例）
$$
\begin{array}{rl}
& \sum\limits_{z_1,z_2,\dots,z_n}\log p(x_1,z_1|\theta)\cdot\prod\limits_{i=1}^np(z_i|x_i,\theta^{(t)})\\
= & \sum\limits_{z_1,z_2,\dots,z_n}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^{(t)})\prod\limits_{i=2}^n p(z_i|x_i,\theta^{(t)}) \\
= & \sum\limits_{z_1}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^{(t)}) \sum\limits_{z_2,\dots,z_n}\prod\limits_{i=2}^n p(z_i|x_i,\theta^{(t)}) \\
= & \left(\sum\limits_{z_1}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^{(t)})\right)\left(\sum\limits_{z_2}p(z_2|x_2,\theta^{(t)})\right)\left(\sum\limits_{z_3}p(z_3|x_3,\theta^{(t)})\right)\dots\left(\sum\limits_{z_n}p(z_n|x_n,\theta^{(t)})\right) \\
= & \sum\limits_{z_1}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^{(t)}) 
\end{array}
$$

所以，将n项加起来，Q函数可简化为
$$
\begin{array}{rl}
Q(\theta, \theta^{(t)}) & = \sum\limits_{z_1}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^{(t)}) + \dots + \sum\limits_{z_n}\log p(x_n,z_n|\theta)\cdot p(z_n|x_n,\theta^{(t)}) \\
 & = \sum\limits_{i=1}^n\sum\limits_{z_i}\log p(x_i,z_i|\theta)\cdot p(z_i|x_i,\theta^{(t)})
\end{array}
$$
回顾一下：
$$
p(X)=\sum\limits_{k=1}^K p_k\cdot N(X|\mu_k, \Sigma_k) \\
p(X,Z) = p(Z)\cdot p(X|Z)=p_z\cdot N(X|\mu_z, \Sigma_z) \\
p(Z|X) = \frac{p(X,Z)}{p(X)} = \frac{p_z N(x|\mu_z,\Sigma_z)}{\sum\limits_{k=1}^Kp_k N(X|\mu_k,\Sigma_k)}
$$

所以Q函数可以继续写为
$$
\begin{array}{rl}
Q(\theta, \theta^{(t)}) & = \sum\limits_{i=1}^n\sum\limits_{z_i}\log p(x_i,z_i|\theta)\cdot p(z_i|x_i,\theta^{(t)}) \\
 & =\sum\limits_{i=1}^n\sum\limits_{z_i}\log p_{z_i}N(x_i|\mu_{z_i},\Sigma_{z_i})\cdot\frac{p^{(t)}_{z_i}\cdot N(x_i|\mu^{(t)}_{z_i},\Sigma^{(t)}_{z_i})}{\sum\limits_{k=1}^K p^{(t)}_k\cdot N(x_i|\mu^{(t)}_k, \Sigma^{(t)}_k)}
\end{array}
$$

我们要求$\theta$其实是$\{p_1,\dots,p_K,\mu_1,\dots,\mu_K, \Sigma_1, \Sigma_K\}$，$\theta^{(t)}$都是已知的。

### M-step

因为$\theta^{(t)}$是已知的，所以$Q(\theta,\theta^{(t)})$的后一项可以认为是已知的，记为$p(z_i|x_i,\theta^{(t)})$. 所以E-step的Q函数可以简写为
$$
\begin{array}{rl}
Q(\theta, \theta^{(t)}) & = \sum\limits_{i=1}^n\sum\limits_{z_i}\log [p_{z_i}N(x_i|\mu_{z_i},\Sigma_{z_i})]\cdot p(z_i|x_i,\theta^{(t)})\\
& = \sum\limits_{z_i}\sum\limits_{i=1}^n\log [p_{z_i}N(x_i|\mu_{z_i},\Sigma_{z_i})]\cdot p(z_i|x_i,\theta^{(t)})\\
& = \sum\limits_{k=1}^K\sum\limits_{i=1}^n\log [p_kN(x_i|\mu_k,\Sigma_k)]\cdot p(z_i=C_k|x_i,\theta^{(t)})\\
& = \sum\limits_{k=1}^K\sum\limits_{i=1}^n[\log p_k + \log N(x_i|\mu_k,\Sigma_k)]\cdot p(z_i=C_k|x_i,\theta^{(t)})
\end{array}
$$
这样就得到了$\theta$，我们即将在M-step优化$\theta$，以$p_k$为例：
$$
\begin{array}{rl}
\max\limits_{p_k} & \sum\limits_{k=1}^K\sum\limits_{i=1}^n\log p_k\cdot
p(z_i=C_k|x_i,\theta^{(t)})\\
s.t. & \sum\limits_{k=1}^K p_k=1
\end{array}
$$
$p_k$是有约束优化，使用拉格朗日乘子法：
$$
L(p,\lambda) = \sum\limits_{k=1}^K\sum\limits_{i=1}^n\log p_k\cdot p(z_i=C_k|x_i,\theta^{(t)}) +\lambda(\sum\limits_{k=1}^K-1)
$$

$$
\begin{array}{rrl}
\frac{\partial L}{\partial p_k} & = \sum\limits_{i=1}^n \frac{1}{p_k}p(z_i=C_k|x_i,\theta^{(t)}) + \lambda & = 0\\
& \sum\limits_{i=1}^n p(z_i=C_k|x_i,\theta^{(t)}) + p_k\lambda & = 0\\
& \sum\limits_{i=1}^n\sum\limits_{k=1}^K p(z_i=C_k|x_i,\theta^{(t)}) + \sum\limits_{k=1}^K p_k\lambda & = 0\\
& n+\lambda & = 0\\
& \lambda & = -n
\end{array}
$$
所以有$p_k^{(t+1)} = \frac{1}{n}\sum\limits_{i=1}^n p(z_i=C_K|x_i, \theta^{(t)})$，$p^{(t+1)} = (p^{(t+1)}_1, \dots, p^{(t+1)}_k)$


# References

1. [机器学习-白板推导系列(十一)-高斯混合模型GMM（Gaussian Mixture Model）](https://www.bilibili.com/video/BV13b411w7Xj)
2. PRML Chapter 9.2


