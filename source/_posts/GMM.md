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

x的概率密度函数：$p(x) = \sum\limits_{Z}p(x,z) = \sum\limits_{k=1}^Kp(x,z=C_k) = \sum\limits_{k=1}^Kp(z=C_k)p(x|z=C_k)=\sum\limits_{k=1}^Kp_kN(x|\mu_k, \Sigma_k)$.  可以发现混合模型角度与几何角度是一致的。

# GMM的求解

## 极大似然无法直接求解

观测数据$X=\{x_1, x_2,\dots, x_n\}$，完整数据$(X,Z)$，参数$\theta=\{p_1,\dots,p_K,\mu_1,\dots,\mu_K, \Sigma_1,\dots,\Sigma_K\}$，如果用极大似然估计直接求解参数
$$
\hat{\theta}_{MLE} = \arg\max\limits_{\theta}\log P(X) = \arg\max\limits_{\theta}\log\prod\limits_{i=1}^N P(X) = \arg\max\limits_{\theta}\sum\limits_{i=1}^N\log P(x_i)
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

含有隐变量Z的混合模型，用EM求解是一种流行且优美的方法。



# References

1. [机器学习-白板推导系列(十一)-高斯混合模型GMM（Gaussian Mixture Model）](https://www.bilibili.com/video/BV13b411w7Xj)