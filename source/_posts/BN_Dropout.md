---
title: Batch Normalization & Dropout
date: 2019-03-17
updated: 2019-03-17
categories:
    - 深度学习
tags:
    - 过拟合
mathjax: true
---

# Batch Normalization & Dropout

from [1]

- **Batch Normalization** Batch Normalization layer can be used in between two convolution layers, or between two dense layers, or even between a convolution and a dense layer. The important question is `Does it help?` Well, it is recommended to use BN layer as it shows improvement generally but the amount of improvement you will get is more problem dependent.
- **Dropout:** Convolution layers, in general, are not prone to overfitting but it doesn't mean that you shouldn't use dropout. You can, but again this is problem dependent. For example, I was trying to build a network where I used `Dropout` in between conv blocks and my model got better with it. It is better if you apply dropout after pooling layer.

from [2]

Dropout works because the process creates multiple implicit ensembles that share weights. The idea is that for each training set, you randomly remove over 50% of the neurons. So effectively, you momentary have a subset of the original neural net that runs inference and gets its weights update. So effective you have many more neural nets working as ensemble to eventually perform the classification.

Batch Normalization is different in that you dynamically normalize the inputs on a per mini-batch basis. The research indicates that when removing Dropout while using Batch Normalization, the effect is much faster learning without a loss in generalization. The research appears to be have been done in Google's inception architecture.

So to answer the question, use Batch Normalization on Inception architectures instead of DropOut. My intuition is that Inception already has a lot of weight sharing going on as a consequence of its optimal structure. Therefore, the generalization benefits of DropOut has diminishing returns.

from [3]

I suggest understanding batch normalization (BN) and dropout separately, though sometime they achieve the same effect/result, which in most case is speeding up training process and prevent from overfitting.

BN mainly focus on lessening the covariate shift. A quick example would be: For a certain pixel position, **layer #1** increase the weight on it by times ten, and then the following layer, **layer #2**, reduce the weight also by ten times (W1 x 10 ==> W2 X 1/10). In the end ,basically these two layers changes covariately thus learns less information. BN helps get rid of above issue and enable each layer learn meaningful information itself by doing mini-batch normalization, while during test, they compute global mean and variance.

Dropout strategy focus on randomly discarding neurons to prevent over-fitting. For example, a dead neuron (the one can not pass activation function thus block information flow) or a over-active neuron (the one split the data space too much) might be removed. A convenient way to look at this: it is a stochastically cross validation strategy. Each time we drop off, a slight different and new architectures come out. In the end, it averages different combinations of the same neural network structure, to get better result. Drop off usually helps when neural network goes quite deep.

In nature, BN and drop-off serves different purpose while in the meantime, they may lead to similar effects. So from this perspective, whether or not use them should depends on your task and network structure. Once formulating your problem properly, you can try them both, either or neither.

**from [4]**

BN跟在activation函数的前面和后面都可以，放在前面比较符合逻辑，但keras是在后面。

Batch size要够大，如果太小的话BN的效果会变差。

好处：

- BN减少了训练时间，可以训练更深的网络，learning rate可以设的大一点

- 减小gradient vanishing和exploding， 适合sigmoid、tanh等激活函数
- 受参数初始化的影响小
- 减少过拟合



References:

1. [Can dropout and batch normalization be applied to convolution layers](https://datascience.stackexchange.com/questions/25722/can-dropout-and-batch-normalization-be-applied-to-convolution-layers)
2. [What is the difference between dropout and batch normalization?](https://www.quora.com/What-is-the-difference-between-dropout-and-batch-normalization)
3. [Should I use a dropout layer if I am using batch normalization in neural network training?](https://www.quora.com/Should-I-use-a-dropout-layer-if-I-am-using-batch-normalization-in-neural-network-training)
4. 深度学习李宏毅视频-Batch Normalization

