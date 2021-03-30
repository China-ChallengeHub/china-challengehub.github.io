+++
title = "图神经网络（03）-Node Embeddings"
date = "2021-03-28"
author = "致Great"
description = "关于 Alert Shortcode 的详细描述"
featured = false
categories = [
  "技术博客"
]
tags = [
  "图神经网络"
]
series = [
  "图神经网络"
]
images = [
]
+++

#我们回归下，如下图所示Graphs用来做传统机器学习任务的流程为：给定输入Graph,然后用来提取节点，边以及图级别的特征，之后学习模型（比如SVM,NN等等），最后将特征映射为标签。
![](https://upload-images.jianshu.io/upload_images/1531909-5467fe93f900ea81.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 1 引言
图表示学习可以减轻特征工程的需求，表示学习可以自动学习Graph的特征，用于下游任务，熟悉NLP的同学比较清楚，传统文本特征构建依赖于统计手段来实现，冗余且效果有限，在词向量Word2Vec和预训练模型Bert等出现之后，文本表示变得更为便利且效果强大，自此万物皆可Embedding。

![](https://upload-images.jianshu.io/upload_images/1531909-fa757f92ac92b88d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图的表示学习的目的就是获得独立于不同任务的高效特征，通俗点讲就是能够针对不同任务学习得到适合任务的嵌入表示。
![](https://upload-images.jianshu.io/upload_images/1531909-b08ec38e84c17c13.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Node Embedding的目的就是能够将节点映射到不同的embedding空间：
- 节点间的embedding的相似性可以表示了节点间在网络的相似性：**如果两个节点之间存在边，那么两个节点越相似**
- 节点的Embedding能够编码网络信息
- 可以用于下游任务，比如节点分类，边的预测，图的分类等
![](https://upload-images.jianshu.io/upload_images/1531909-2f0fa197e3681746.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
下面是一个Zachary's Karate Club Network的2维节点Embedding展示：
![](https://upload-images.jianshu.io/upload_images/1531909-73f025e2771ffe23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 2 节点嵌入：编码和解码
这一节我们主要掌握下如何学习节点的嵌入向量
### 2.1 构建输入-图
首先，假设我们有一个图$G$：
- $V$代表节点的集合
- $A$ 代表链接矩阵
为了方便起见，我们默认认为节点没有其他的额外特征，如下图所示：
![](https://upload-images.jianshu.io/upload_images/1531909-cdbd8c34f5b5d15a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 2.2 学习目标-Node Embedding
我们的目标是能够学习到节点的嵌入表示，这种节点嵌入的相似性能够近似节点在图中的相似性。
![](https://upload-images.jianshu.io/upload_images/1531909-95dbf6f9cdcb925f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图中$ENC(u)$代表对节点$u$进行编码表示得到$z_{u}$,同样$ENC(v)$代表对节点$u$进行编码表示得到$z_{v}$

### 2.3 学习方法-Encoder和Decoder
1. **`Encoder`**能够将节点映射成向量
2. 定义一个节点相似性函数(比如节点在原始图中的相似性评估方法)
3. Decoder `DEC`能够将节点向量映射成相似性分数
4. 优化编码器的参数：
原始网络的相似性：$similarity(u,v)$  $\approx$  节点嵌入的相似性 ： $z_{v}^Tz_{u}$

其中有两个非常关键因素：编码和相似性函数。编码器能够学习到节点的向量，相似性函数能够提供我们优化的目标，如果两个节点在图中存在边或者距离越近，那么它们对应的节点向量在向量空间中相似性也会越大。
![](https://upload-images.jianshu.io/upload_images/1531909-d0c40659b5c62f3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如何选择定义节点相似性的方法是关键的地方，我们考虑下什么情况下，两个节点的相似性比较高？
- 是否存在边或者连接
- 是否共享邻居节点
- 是否包括相似的属性特征
接下来我们通过随机游走（Random Walks）来学习节点相似性，以及优化节点的嵌入向量。

## 3 Random Walk
Node Embeddings的计算方法之一就是Random Walk，为了方面起见我们定义和引出下面的符号，以便统一解释。
### 3.1 符号
为了方面接下来的公式推导，我们统一下符号标记：
- 向量 $z_{u}$:
   > u节点的嵌入向量
- 概率$P(v|z_{u})$：
  > 从节点$u$出发通过随机游走访问节点$v$的概率
- Softmax函数：通过softmax函数一作用，一个K维向量就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标。
$\sigma(z)_{i}=\frac{e^{Z_{i}}}{\sum_{j=1}^Ke^{Z_{j}}}$
- Sigmoid函数：
$S(x)=\frac{1}{1+e^{-x}}$将一个数映射到（0,1）的区间
### 3.2 Random Walk 定义
给定一个Graph和一个起始点，我们选择随机选择该点的邻居节点，并且移动到该邻居节点；然后我们根据当前节点随机选择一个邻居节点，然后移动到该邻居节点，重复上述步骤...通过这种随机游走访问Graph节点的方式可以产生随机序列。下图描述了一个随机游走的实例，第一步从起始节点出发移动到节点5，第二步从节点5移动节点8，第三步从节点8移动节点9，第四步又从节点9移动到节点8，之后第5步移动到节点11，这样访问路径构成一个序列：
```
start_node->5->8->9->8->11
```
![](https://upload-images.jianshu.io/upload_images/1531909-5704af770d20a98c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 3.3 Random Walk如何得到Node Embeddings
相信大家对下面的公式不会感到陌生，一般我们衡量两个向量的相似性时可以通过两个向量点积计算得到，相似地两个节点同时出现在同一随机游走序列的概率可以用如下表示：
![](https://upload-images.jianshu.io/upload_images/1531909-1b9b91feebbd22b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
所以我们通过Rankdom Walk得到Node Embedding可以通过两个步骤：
(1) 通过某种随机游走策略$R$（下文将会介绍）得到从节点$u$出发访问到节点$v$的路径，然后可以估计出节点$u$和$v$同时出现的概率
(2) 优化节点的embeddings表示，使得节点的嵌入表示能够表达或者近似我们上述随机游走得到的统计
回想一下我们机器学习或者深度学习的各种任务，无非就是在优化一个目标，一个近似目标，我们可以笼统地将我们输入表示成X，然后通过方法f来近似y。说到这里上面两个步骤特别像NLP中Glove向量的优化步骤，首先我们统计两个词在语料中共现概率，然后通过词向量来近似共现概率。

![](https://upload-images.jianshu.io/upload_images/1531909-2af48cdb61b11d9d.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)








