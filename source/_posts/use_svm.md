---
title: SVM的使用
date: 2018-10-23
updated: 2018-11-18
categories:
    - 机器学习
tags:
    - SVM
mathjax: true
---

正如Andrew Ng所说，使用svm进行训练，相比神经网络来说，确实训练速度要快很多。
不过只有当特征数量 n 很小，训练集数据量$m$中等的时候，才建议使用高斯核函数来训练SVM，例如$n$的范围是$( 1,1000 )$，$m$的范围是$( 10,10000 )$。
如果特征数量$n$远远大于训练样本集$m$，或者特征数量$n$很小，而训练样本集$m$很大，都不宜使用高斯核函数。

连接：https://zhuanlan.zhihu.com/p/30073946

一般用线性核和高斯核，也就是Linear核与RBF核需要注意的是需要对数据归一化处理，很多使用者忘了这个小细节然后一般情况下RBF效果是不会差于Linear但是时间上RBF会耗费更多，其他同学也解释过了下面是吴恩达的见解：1. 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel3. 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

链接：https://www.zhihu.com/question/21883548/answer/112128499

__SVM如何防止过拟合？__
- 减小参数C
- 增加软间隔
