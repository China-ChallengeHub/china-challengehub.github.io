+++
title = "【比赛推送】KDD 2021/ Kaggle 顶级赛事等你来排位"
date = "2021-03-20"
description = "关于 Alert Shortcode 的详细描述"
featured = false
categories = [
  "比赛推送"
]
tags = [
  "比赛推送"
]
series = [
  "比赛推送"
]
images = [
]
+++

# KDD 2021 赛题
### 赛题1 基于多数据集的时间序列异常检测
> Multi-dataset Time Series Anomaly Detection
https://compete.hexagon-ml.com/practice/competition/39/

![](https://upload-images.jianshu.io/upload_images/1531909-132e47c1edbc8795.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 背景描述-时间序列异常检测
近年来，SIGKDD以及其他数据挖掘，机器学习和数据库会议上出现了有关时间序列异常检测的论文。这些论文中的大多数都在一个或多个基准数据集中进行测试，包括由NASA，Yahoo，Numenta和清华-OMNI 等创建的数据集。

尽管社区应该非常赞赏这些团队共享数据的努力，但最近的几篇论文[a]认为这些数据集不适用于衡量异常检测的进展。

简而言之，反对使用这些数据集的两个最引人注目的论据是：

**冗余性**：几乎所有上述基准数据集都可以完美解决，而无需查看任何训练数据，并且使用已有十年历史的算法。
**标签错误**：永远不能完全消除错误检测基准错误贴标签的可能性。但是，上面提到的某些数据集在基本事实中似乎有大量假阳性和假阴性。已经发表了一些论文，认为方法A比方法B更好，因为它在基准X上的准确性要高5％。但是，仔细检查基准X可以发现，有25％以上的标签是错误的，这个数字使标签A的准确性相形见war。声称所比较算法之间的差异。

除了上面列出的问题以及文件重叠的可能性外，我们认为社区还存在一系列不合适的基准。考虑到这一点，作为比赛的一部分，我们为时间序列异常检测创建了新的基准。

为此竞赛创建的基准数据集旨在缓解此问题。重要的是要注意我们的主张是“缓解”，而不是“解决”。我们认为，非常有很多研究者本着CASP [d]的精神来解决这个问题将是很棒的。

同时，作为这一挑战的一部分的200个数据集反映了20多年研究时间序列异常检测文献并收集数据集的工作。除了这场竞赛的本身，我们希望它们可以在未来几年中为社区提供资源，并激发对异常检测评估的更深刻的思考。

### 赛题奖励
- 一等奖：$ 2000 USD
- 二等奖：$ 1000 USD
- 三等奖：$ 500美元
- 对于排名前15位的参与者，我们将提供具有等级的证书。
- 对于所有其他参与者，我们将提供参赛证书（阳光普照奖）
### 赛题时间轴
- 阶段1:2021年3月15日-2021年4月7日
- 阶段2:2021年4月7日-20210年6月1日
### 赛题任务与数据
赛题任务：预测时间序列中异常发生的位置
## 数据

| 姓名 | 下载 |
|----|----|
| 提交样例 | ```[下载](https://compete.hexagon-ml.com/media/data/multi-dataset-time-series-anomaly-detection-39/submissionsample.csv) ```|
| 数据 | ```[下载](https://compete.hexagon-ml.com/media/data/multi-dataset-time-series-anomaly-detection-39/data.zip)``` |

文件格式

这些文件使用命名约定，该约定在测试和训练之间提供了分隔。

<id> _ <名称> _ <拆分号> .txt

例如004_UCR_Anomaly_2500.txt。 

此处split-number = 2500表示从2500开始存在异常。

![image](https://upload-images.jianshu.io/upload_images/1531909-55764c11b4f8860e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

样本提交文件

![image.png](https://upload-images.jianshu.io/upload_images/1531909-6cb4397eb8aa9b9b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

提示：

提交文件的column标头应“完全”匹配预期的格式。例如，编号，位置

行数应与总计数完全匹配（第一阶段：25行，第二阶段：200行）

**location的值是一个整数。**

第一阶段 

数据部分将提供25个时间序列文件以及示例提交文件。 

这将是一个培训阶段，在进入第二阶段时将清除排行榜。

第二阶段

比赛第二阶段将提供200个时间序列文件，包括第一阶段的前25个文件。

### 提交评估
我们在异常范围的两侧添加了+/- 100个位置，以奖励正确的答案。

**例子**

![](https://upload-images.jianshu.io/upload_images/1531909-83382932f13a5779.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**有200个文件，对于每个正确的答案，您将获得1分，对于错误的答案将获得0分。您可以获得的最高分数是100％（只要您使用算法在代码中执行此操作，就无需人工标记）。如果我们怀疑有任何参与者**

赛题可以提交了，别等待了~
![](https://upload-images.jianshu.io/upload_images/1531909-1a3c51601788fbca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 赛题2 城市大脑挑战-交通网络调度
> City Brain Challenge
http://www.yunqiacademy.org/

![](https://upload-images.jianshu.io/upload_images/1531909-f16101ba871d93bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 赛题背景
没有人喜欢被卡在城市交通中。尽管我们在城市中观察到许多车辆，但交通拥堵的原因仍不清楚。是因为车辆数量超出了城市的承载能力，还是因为我们未能以最大承载能力利用道路网络？

以世界上最大的两个城市为例。东京和纽约市的交通拥堵指数排名相似。但是，值得注意的是，东京的注册车辆比纽约市多50％，而东京的信号交叉口仅比纽约市多15％，道路长度比纽约多30％。（东京：[注册车辆313万，](https://dmv.ny.gov/statistic/2018reginforce-web.pdf)[交通信号15,000 ](http://www.utrc2.org/sites/default/files/Final-Report-Develop-Comprehensive-Guide-Traffic-Signal-Timing.pdf)[，道路24,650公里。](https://www.dot.ny.gov/divisions/engineering/technical-services/hds-respository/Tab/NYSDOT_Highway_Mileage_Summaries_2017.pdf)纽约市[注册车辆219万，](https://www.statista.com/statistics/1191244/japan-number-motor-vehicles-in-use-tokyo/#:~:text=Number%20of%20registered%20motor%20vehicles%20in%20Tokyo%2C%20Japan%2C%202010%2D2019&text=As%20of%20March%2031%2C%202019,3.29%20million%20vehicles%20in%202010)[交通信号13,000 ](https://trid.trb.org/view/1215582)[，道路18,684 km。](https://www.statista.com/statistics/1179384/japan-tokyo-road-length/)）[](https://dmv.ny.gov/statistic/2018reginforce-web.pdf)[](http://www.utrc2.org/sites/default/files/Final-Report-Develop-Comprehensive-Guide-Traffic-Signal-Timing.pdf)[](https://www.dot.ny.gov/divisions/engineering/technical-services/hds-respository/Tab/NYSDOT_Highway_Mileage_Summaries_2017.pdf)[](https://www.statista.com/statistics/1191244/japan-number-motor-vehicles-in-use-tokyo/#:~:text=Number%20of%20registered%20motor%20vehicles%20in%20Tokyo%2C%20Japan%2C%202010%2D2019&text=As%20of%20March%2031%2C%202019,3.29%20million%20vehicles%20in%202010)[](https://trid.trb.org/view/1215582)[](https://www.statista.com/statistics/1179384/japan-tokyo-road-length/)

为什么东京可以提供比纽约更多的车辆服务？纽约市是否以最大载客量运营交通？作为数据科学家，我们邀请您协调交通，并根据城市规模的路网及其交通需求找到最大交通容量。

### 比赛描述

在这一挑战中，我们将为您提供一个城市规模的道路网络，其交通需求来自真实的交通数据。您将负责协调信号交叉口的交通，同时将延迟指数保持在预定义的阈值以下。我们将增加流量需求，并查看您的协调模型是否仍然可以凑效。

为了促进您的方法开发，我们将首次发布City Brain Open Research Platform。该平台包含一个城市规模的交通模拟环境和一个具有多核计算机的云计算集群。
![](https://upload-images.jianshu.io/upload_images/1531909-d4ff845fe143704b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 时间线
- 报名 4/1/2021

参赛选手熟悉区域交通的数据以及熟悉模拟环境。

- 参赛 5/1/2021

参与者将进行城市规模的交通协调。可以处理更大流量需求的团队将进入最后一轮。
- 最终提交 6/1/2021

提供大规模的云计算平台。团队将开发方法来处理城市范围内各种未知的交通流量。


- 比赛结束 7/1/2021

### 比赛奖励
奖金：
- 第一名：$ 3000
- 第二名：$ 2000
- 第三名：$ 1000
- 第四至第十名：$ 500

研究支持：
- 大规模计算资源
- 潜在的现实世界级研究实验
- 纸质出版物的其他支持

### 赛题3 大型图机器学习比赛： OGB-LSC
>OGB Large-Scale Challenge 
https://ogb.stanford.edu/kddcup2021/

![](https://upload-images.jianshu.io/upload_images/1531909-e4656930d8122c07.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 为什么要举行大型图ML比赛？

由于在实际应用中普遍使用图结构化数据，因此图上的机器学习（ML）近年来引起了极大的关注。现代应用领域包括网络规模的社交网络，推荐系统，超链接的网络文档，知识图谱（KGs），以及通过不断增长的科学计算生成的分子模拟数据。这些域涉及具有数十亿个边的大规模图形或具有数百万个图形的数据集。大规模部署准确的图ML将产生巨大的实际影响，从而实现更好的推荐结果，改进的Web文档搜索，更全面的KG以及基于ML的准确药物和材料发现。

然而，社区在大规模图形ML中发展最新技术的努力非常有限。实际上，处理大规模图具有挑战性，特别是对于最先进的表达性图神经网络（GNN），因为它们会根据来自许多其他节点的信息对每个节点进行预测。要有效地大规模训练这些模型，就需要复杂的算法，而这些算法远远超出了基于iid数据的标准SGD。最近，研究人员通过显着简化GNN来提高模型可伸缩性，这不可避免地限制了它们的表达能力。

但是，在深度学习中，一遍又一遍地表明，人们需要大型的表达模型并在大数据上对其进行训练，以实现最佳性能。在图ML中，趋势是相反的-模型变得简化且表达能力较弱，因此无法缩放到大图。因此，存在着巨大的机会来移动社区以使用现实的和大规模的图形数据集，并将领域的状态向前移动到需要的地方。

### OGB-LSC概述
在这里，我们提出了一个大型图ML竞赛，即**OGB大型挑战赛（OGB-LSC）**，以鼓励开发适用于海量现代数据集的最新图ML模型。具体来说，**我们提供了三个数据集：[MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/)**，**[WikiKG90M-LSC](https://ogb.stanford.edu/kddcup2021/wikikg90m/)**和**[PCQM4M-LSC](https://ogb.stanford.edu/kddcup2021/pcqm4m/)**，它们的规模空前大，并且分别覆盖了节点，链接和图形级别的预测。 **每个数据集提供一个独立的任务，优胜者将分别为每个数据集选择。** 我们将宣布每个数据集的前3名获胜团队（总共9个获胜团队），他们将有机会在KDD Cup研讨会上展示他们的解决方案。
下面提供了三个OGB-LSC数据集的说明性概述：
![](https://upload-images.jianshu.io/upload_images/1531909-1059be647bb7398e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   **[MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/)**是一个异构的学术图，其任务是预测位于异构图中的论文的主题区域（节点分类）。
*   **[WikiKG90M-LSC](https://ogb.stanford.edu/kddcup2021/wikikg90m/)**是一个知识图，其任务是估算缺少的三元组（链接预测）。
*   **[PCQM4M-LSC](https://ogb.stanford.edu/kddcup2021/pcqm4m/)**是量子化学数据集，其任务是预测给定分子的重要分子特性，即HOMO-LUMO间隙（图形回归）。

对于每个数据集，我们都会仔细设计其预测任务和数据拆分，以便在任务上实现较高的预测性能将直接影响相应的应用程序。每个数据集页面中都提供了更多详细信息。

数据集统计信息和基本信息总结如下，表明我们的数据集非常大。
![](https://upload-images.jianshu.io/upload_images/1531909-179141cc606b4e3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**所有这些数据集都可以使用我们的[`ogb`Python包](https://github.com/snap-stanford/ogb)下载并准备。** **模型评估和测试提交文件的准备工作也由我们的软件包处理。** 用法在每个数据集页面中都有描述。请通过以下方式安装/更新：

在我们的**[论文中](http://arxiv.org/abs/2103.09430)**，我们进一步对每个数据集进行了广泛的基线分析，大规模实现了简单的基线模型以及高级的表达模型。我们发现，尽管需要更多的努力来进行扩展，但先进的表达模型确实会从大数据中受益，并且明显优于易于扩展的简单基线模型。我们所有的基准代码均已**[公开提供，](https://github.com/snap-stanford/ogb/tree/master/examples/lsc)**以方便公众研究。
```
pip install -U ogb
# Make sure below prints the required package version for the dataset you are working on.
python -c "import ogb; print(ogb.__version__)"
```
总体而言，我们的KDD杯将鼓励社区开发和扩展表达性图ML模型，这可以在各个领域取得重大突破。我们希望在2021年KDD杯上的OGB-LSC能够成为图ML领域的“ ImageNet大规模视觉识别挑战”，鼓励社区致力于现实和大规模的图数据集，并显着提高现状-艺术。
**[OGB-LSC数据集论文](http://arxiv.org/abs/2103.09430)**


# Kaggle新赛题
## 赛题1 Shopee - Price Match Guarantee
通过两个图像确定两个产品是否相同
> https://www.kaggle.com/c/shopee-product-matching/overview/description

![](https://upload-images.jianshu.io/upload_images/1531909-832a551d993f124c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 评估指标
采用F1-Score，参赛作品将根据其平均[F1分数](https://en.wikipedia.org/wiki/F1_score)进行评估。均值以样本方式计算，这意味着将为每个预测行计算F1分数，然后取平均值。

选手必须创建一个以空格分隔的列表，`posting_id`该列表与该`posting_id`列中的发布匹配的所有。*帖子总是自我匹配的*。匹配个数上限为50，因此预测50场以上的对比赛没有好处。

该文件应有一个标题，名为`submission.csv`，如下所示：
```
posting_id,matches
test_123,test_123
test_456,test_456 test_789

```
您应该预测每个比赛`posting_id`。例如，如果您认为A与B和C相匹配，则`A,A B C`还应包含`B,B A C`和`C,C A B`。

### 时间线
- 2021 年3月8日-比赛开始日期
- 2021 年5月3日-报名截止日期。您必须在此日期之前接受比赛规则才能参加比赛。
- 2021 年5月3日-合并团队截止日期。这是参与者可以加入或合并团队的最后一天。
- 2021 年5月10日-最终提交截止日期。

### 奖金
- 第一名-$ 15,000
- 第二名-$ 10,000
- 第三名-$ 5,000

### 数据集
在大型数据集中查找近重复项是许多在线业务的重要问题。对于Shopee，日常用户可以上传自己的图像并撰写自己的产品说明，从而增加了挑战。您的任务是确定哪些产品已重复发布。相关产品之间的差异可能微妙，而相同产品的照片则可能千差万别！

由于这是一场代码竞赛，因此仅发布测试集的前几行/图像；其余的仅在提交笔记本后才对您的笔记本可用。预计将在隐藏的测试集中找到大约70,000张图像。提供的一些测试行和图像旨在说明隐藏的测试集格式和文件夹结构。

![](https://upload-images.jianshu.io/upload_images/1531909-dd9bd487d61f0e73.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- **[train / test] .csv-**训练集数据。每行包含一次过帐的数据。多个订单可能具有完全相同的图像ID，但标题不同，反之亦然。

*   `posting_id` -发布的ID码。

*   `image` -图片ID / md5sum。

*   `image_phash`-图像的[感知哈希](https://en.wikipedia.org/wiki/Perceptual_hashing)。

*   `title` -发布的产品说明。

*   `label_group`-映射到同一产品的所有发布的ID码。未提供测试集。

**[train / test]** images-与发布相关的图像。

**sample_submission.csv-**格式正确的示例提交文件。

*   `posting_id` -发布的ID码。

*   `matches`-以空格分隔的与此发布匹配的所有发布ID的列表。posting是自我匹配，匹配个数上限为50，因此无需预测超过50个数。

## 赛题2 Bristol-Myers Squibb – Molecular Translation
化学分子图像转换为化学结构序列
https://www.kaggle.com/c/bms-molecular-translation

![](https://upload-images.jianshu.io/upload_images/1531909-3f80569a0847a519.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 背景描述
在技术进步的世界中，有时最好，最简单的工具仍然是纸和笔。有机化学家经常利用骨架式（Skeletal formula）来绘制分子工作，骨架式是一个世纪以来使用的结构符号。最近的出版物还用机器可读的化学描述（InChI）进行了注释，但是数十年的扫描文档无法自动搜索特定的化学描述。在机器学习的帮助下，光学化学结构的自动识别可以加快研发的速度。

不幸的是，大多数公共数据集太小，无法支持现代机器学习模型。现有工具只能在最佳条件下产生90％的精度。历史来源通常会在某种程度上破坏图像，从而将性能降低到接近零。在这些情况下，需要耗时的手动操作才能将扫描的化学结构图像可靠地转换为机器可读格式。

百时美施贵宝公司是一家全球性生物制药公司，致力于通过科学改变患者的生活。他们的任务是发现，开发和提供可帮助患者战胜严重疾病的创新药物。

在这场比赛中，您将解释旧的化学图像。通过访问由Bristol-Myers Squibb生成的大量合成图像数据，您可以将图像转换回标注为InChI文本的基础化学结构。

整理化学文献的工具将对研究人员带来重大好处。如果成功，您将帮助化学家扩大进行集体化学研究的机会。反过来，通过避免重复先前发表的化学方法，并通过挖掘大数据集来识别新趋势，这将加快许多关键领域的研发工作。

### 评估指标
根据您提交的InChi字符串与地面真实InChi值之间的平均[Levenshtein距离](http://en.wikipedia.org/wiki/Levenshtein_distance)评估提交的内容。

## 提交文件

对于`image_id`测试集中的每一个，您必须在相应的图像中预测分子的InChi字符串。该文件应包含标头，并具有以下格式：

```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
000037687605,InChI=1S/H2O/h1H2
etc.
```
### 时间轴
- 2021 年3月2日-比赛开始日期
- 2021 年5月26日-报名截止日期。您必须在此日期之前接受比赛规则才能参加比赛。
- 2021 年5月26日-合并团队截止日期。这是参与者可以加入或合并团队的最后一天。
- 2021 年6月2日-最终提交截止日期。
### 奖金
- 第一名-$ 25,000
- 第二名-$ 15,000
- 第三名-$ 10,000

### 数据集
在本次比赛中，您将获得化学品的图像，目的是预测图像中相应的[国际化学品识别码](https://en.wikipedia.org/wiki/International_Chemical_Identifier)（InChI）文本字符串。所提供的图像（训练数据和测试数据中的图像）都可以旋转到不同的角度，具有不同的分辨率，并且具有不同的噪声水平。

**注意：**此数据集中总共约有4m张图像。解压缩下载的数据将花费时间。

- train / -训练图像，由3级文件夹结构排列image_id
![](https://upload-images.jianshu.io/upload_images/1531909-179f920365f2ee41.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- test / -测试图像，与文件夹位于相同的文件夹结构中train/
- train_labels.csv-训练图像的地面真相InChi标签
- sample_submission.csv-格式正确的样本提交文件