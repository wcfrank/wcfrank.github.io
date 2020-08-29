# VAE介绍

VAE也是latent variable model. 如果GMM是K个高斯分布的混合，那么VAE是无限多个高斯分布的混合。高斯分布只有K个高斯分布的混合$P(Z)=p_k$，无法处理复杂的任务，只能做简单的聚类。

VAE的高斯分布$Z\sim N(0,I)$是连续的+高维的高斯分布。如果X是连续的，$X|Z\sim N(\mu_{\theta}(Z), \Sigma_{\theta}(Z))$，这里$\mu, \Sigma$不再是常数，而是关于$\theta$的函数，这样的目的是用神经网络去逼近概率分布。
$$
p_{\theta}(X) = \int_{Z}p_{\theta}(X,Z) = \int_Zp(Z)p_{\theta}(X|Z)dZ
$$
因为Z是高维的，所以上面的积分，即$p(X)$是intractable，所以后验$p_{\theta}(Z|X)=p(Z)P_{\theta}(X|Z)/p_{\theta}(X)$是intractable. 

# VAE的learning、inference问题

如果用EM算法的公式来求$q_{\theta}(Z|X)$：
$$
\log p(X) = ELBO + KL(q_{\phi}(Z|X), p_{\theta}(Z|X))
$$

- E-step： 当$q=p_{\theta}(Z|X)$时，KL divergence=0，Expectation is ELBO 
- M-step: $\theta = \arg\max\limits_{\theta} ELBO = \arg\max\limits_{\theta} E_{p_{\theta'}(Z|X)}[\log p_{\theta}(X, Z)]$

但VAE无法使用EM算法来解决，因为EM算法的前提是后验概率$p_{\theta}(Z|X)$为tractable的。但因为$p_{\theta}(Z|X)$是intractable，我们用$q_{\phi}(Z|X)$来逼近它。那就只能找到一个q来逼近后验概率：
$$
\begin{array}{rl}
<\hat{\theta}, \hat{\phi}>&=arg\min\limits_{}KL(q_{\phi}(Z|X), p_{\theta}(Z|X)) \\
& = arg\max ELBO\\
& = \arg\max E_{q_{\phi}(Z|X)}[\log p_{\theta}(X,Z)] + H[q_{\phi}]\\
& = \arg\max E_{q_{\phi}(Z|X)}[\log p_{\theta}(X|Z)] - KL(q_{\phi}(Z|X)||p(Z)) 
\end{array}
$$

上式对$\theta$或$\phi$求梯度，就可以得到想要的参数值，具体的过程也参考变分推断的求解。

一般假设$Z|X\sim N(\mu_{\phi}(X), \Sigma_{\phi}(X))$，经过重参数化技巧，$Z = \mu_{\phi}(X) + \Sigma_{\phi}^{1/2}(X)\cdot\epsilon$.  

考虑公式里的$\log p_{\theta}(X|Z)$，是给定Z的情况下看X，是decoder；Z是来自于$q_{\phi}(Z|X)$，是encoder。在learning过程中，先从X到Z，然后再从Z到X，就是从encoder到decoder的过程。公式里的第二项KL divergence是正则化项。


# References

1. [【机器学习】白板推导系列(三十二) ～ 变分自编码器(VAE)](<https://www.bilibili.com/video/BV15E411w7Pz?from=search&seid=9357319156757457518>)
2. [NNDL](https://nndl.github.io/nndl-book.pdf)

