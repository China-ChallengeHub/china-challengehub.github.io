+++
title = "图神经网络（04）-基于Graph的传统机器学习方法"
date = "2021-03-29"
publishDate = 2021-03-29
description = "图神经网络（04）-基于Graph的传统机器学习方法"
featured = false
draft = false
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

## 1 图学习任务
我们简单回顾下，上一节我们介绍了，图的机器学习任务主要是以下三种：
- Node Level:节点级别
- Link Level：边级别
- Graph Level：图级别å
并且三部分难度依次是由浅入深的
![](https://upload-images.jianshu.io/upload_images/1531909-abad0a586f582952.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 2 传统ML流程
- 定义和设计节点/边/图的特征
-  对所有训练数据构造特征
![](https://upload-images.jianshu.io/upload_images/1531909-2585e8268d1c5936.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 训练ML模型
（1）随机森林
（2）支持向量机
（3）神经网络等
- 应用模型
给定一个新的节点、边、图，然后获取特征进行预测

![](https://upload-images.jianshu.io/upload_images/1531909-4223b78ff1e08202.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们总结下 基于Graph的机器学习相关概念和流程，首先明确下目标

目标：对一些对象集合进行预测，比如是分类或者回归任务
特征设计：
- 特征:`d-dimensional`向量
- 对象：Nodes，edges，或者是graps
- 目标函数：结合具体任务设计目标函数，如下图所示给定一个图G(V,E)，其中V代表节点集合，E代表边集合，然后学习节点到向量空间R的映射函数，也就是我们要学习权重参数W
![](https://upload-images.jianshu.io/upload_images/1531909-c2933b8e984a4dec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 为了方便，我们下面的例子是基于无向图(undirected grpah)进行解释的。

## 3 节点级别的相关任务
基于图中带有标签的节点训练模型，然后预测未标注节点的标签，
![](https://upload-images.jianshu.io/upload_images/1531909-b3c32497a27631e2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在这里我们主要阐述下Node的四种特征：
- Node degree:节点的度
- Node centrality:节点的中度
- Clustering coefficient:相似性
- Graphlets:图元
![](https://upload-images.jianshu.io/upload_images/1531909-372ff242e0cb001e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**节点的度**
-  k<sub>v</sub>代表是节点v与邻居节点相连边的个数
- 所有邻居节点都是相等的

如下图所示，A的度为1，B的度为2，C的度为3，D的度为4
![](https://upload-images.jianshu.io/upload_images/1531909-fe6758c41894aa99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**节点的中心度  Node Centrality**
- 节点的度只计算了相连节点的个数，但是没有评估节点的重要性
- 节点的中心度$c_v$考虑了节点在图中的重要程度
- 节点的中心度有很多种计算方式：
(1) Engienvector centrality:特征向量中心性
(2) Betweenness centrality:中介中心性
(3) Closeness centrality:紧密中心性
(4) 其他方式

**Eigenvector centrality:特征向量中心性**
- 如果一个节点的邻居节点们$u\in N(v)$越重要，那么该节点$v$就越重要
- 我们将节点𝑣的中心性建模为相邻节点的中心性之和：
$c_v=\frac{1}{\lambda}\sum_{u \in N(v)}{c_u}$
其中$\lambda$为一个正常数。
- 我们注意到上面的等式是通过递归的方式来计算度度中心性的，所有我们应该怎么求解

$c_v=\frac{1}{\lambda}\sum_{u \in N(v)}{c_u} \iff \lambda c=Ac$

其中$\lambda$为正常数，$A$为邻接矩阵，如果$u \in N(v)$ ,那么$A_{uv}=1$

- 我们可以看到度中心性是一个特征向量(eigenvector)，根据非负矩阵定理(Perron-Frobenius Theorem)我们可以知道$\lambda_{max}$是一个正值并且是唯一的，特征向量$c_{max}$当做度中心性。

通常来说，有许多不同的特征值$\lambda$ 能使得一个特征方程有非零解存在。然而，考虑到特征向量中的所有项均为非负值，根据佩伦-弗罗贝尼乌斯定理，只有特征值最大时才能测量出想要的中心性。然后通过计算网络中的节点$v$其特征向量的相关分量$v^{th}$便能得出其对应的中心性的分数。

> 特征向量的定义只有一个公因子，因此各节点中心性的比例可以很好确定。为了确定一个绝对分数，必须将其中一个特征值标准化，例如所有节点评分之和为1或者节点数 n。幂次迭代是许多特征值算法中的一种，该算法可以用来寻找这种主导特征向量。此外，以上方法可以推广，使得矩阵A中每个元素可以是表示连接强度的实数，例如随机矩阵。---特征向量中心性wiki


这里其实涉及到比较多的线性代数的理论以及矩阵分析的算法，我特意查了一些文章帮大家去回顾和理解下这里涉及到的知识：
(1) [线性代数之——特征值和特征向量](https://zhuanlan.zhihu.com/p/51011531)
(2)[非负矩阵之Perron-Frobenius定理](https://zhuanlan.zhihu.com/p/80952693)
(3)[非负矩阵](https://wenku.baidu.com/view/b9d7f70216fc700abb68fceb.html)
(4)[干货 | 万字长文带你复习线性代数！
](https://zhuanlan.zhihu.com/p/52013163)

我们这里再通过文章[谁是社会网络中最重要的人？](https://zhuanlan.zhihu.com/p/31198752)解释特征向量中心性：
特征向量中心性的基本思想是，一个节点的中心性是相邻节点中心性的函数。也就是说，**与你连接的人越重要，你也就越重要**。

特征向量中心性和点度中心性不同，一个点度中心性高即拥有很多连接的节点特征向量中心性不一定高，因为所有的连接者有可能特征向量中心性很低。同理，特征向量中心性高并不意味着它的点度中心性高，它拥有很少但很重要的连接者也可以拥有高特征向量中心性。

考虑下面的图，以及相应的5x5的邻接矩阵(Adjacency Matrix)，A。

![](https://upload-images.jianshu.io/upload_images/1531909-6a6ef214f22662f4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

邻接矩阵的含义是，如果两个节点没有直接连接，记为0，否则记为1。

现在考虑x，一个5x1的向量，向量的值对应图中的每个点。在这种情况下，我们计算的是每个点的点度中心性（degree centrality），即以点的连接数来衡量中心性的高低。

矩阵A乘以这个向量的结果是一个5x1的向量：

![](https://upload-images.jianshu.io/upload_images/1531909-a91c3832858f21b3.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

结果向量的第一个元素是用矩阵A的第一行去“获取”每一个与第一个点有连接的点的值（连接数，点度中心性），也就是第2个、第3个和第4个点的值，然后将它们加起来。

换句话说，邻接矩阵做的事情是将相邻节点的求和值重新分配给每个点。这样做的结果就是“扩散了”点度中心性。你的朋友的朋友越多，你的特征向量中心性就越高。

![](https://upload-images.jianshu.io/upload_images/1531909-e444f015f2f1eaa5.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们继续用矩阵A乘以结果向量。如何理解呢？实际上，我们允许这一中心性数值再次沿着图的边界“扩散”。我们会观察到两个方向上的扩散（点既给予也收获相邻节点）。我们猜测，这一过程最后会达到一个平衡，特定点收获的数量会和它给予相邻节点的数量取得平衡。既然我们仅仅是累加，数值会越来越大，但我们最终会到达一个点，各个节点在整体中的比例会保持稳定。

现在把所有点的数值构成的向量用更一般的形式表示：

![](https://upload-images.jianshu.io/upload_images/1531909-faabf43bad5dc54f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们认为，图中的点存在一个数值集合，对于它，用矩阵A去乘不会改变向量各个数值的相对大小。也就是说，它的数值会变大，但乘以的是同一个因子。用数学符号表示就是：

![](https://upload-images.jianshu.io/upload_images/1531909-58b92183dc2b2b79.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

满足这一属性的向量就是矩阵M的特征向量。特征向量的元素就是图中每个点的特征向量中心性。

特征向量中心性的计算需要读者具备矩阵乘法和特征向量的知识，但不影响这里读者对特征向量中心性思想的理解，不再赘述。

**Betweenness centrality:中介中心性**
 如果一个节点位于很多条其他节点的最短路径上，那么改节点比较重要，定义如下：
$c_{v}=\sum_{s\not=v\not=t}{\frac{\#(shortes\,paths \,betwen \,`𝑠 `\, and \, `𝑡` \, that \,contain \,`𝑣`)}{\#(shortest \,paths \,between \,'𝑠' \,and \,'𝑡')}}$
我们可以看一个下面的例子：
![](https://upload-images.jianshu.io/upload_images/1531909-d539998922c21800.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设我们要计算D的中介中心性：
- 首先，我们计算节点D之外，所有节点对之间的最短路径有多少条，这里是1条（在5个节点中选择两个节点即节点对的个数）。
- 然后，我们再看所有这些最短路径中有多少条经过节点D，例如节点A要想找到节点E，必须经过节点D。经过节点D的最短路径有3条（A-C-D-E，B-D-E，C-D-E）。
- 最后，我们用经过节点D的最短路径除以所有节点对的最短路径总数，这个比率就是节点D的中介中心性。节点D的中介中心性是3/1=3。

## 4 PyTorch Geometric教程
官方文档：`https://pytorch-geometric.readthedocs.io/en/latest/index.html`

PyTorch Geometric（PyG）是PyTorch的扩展库。 它提供了有用的框架来开发图深度学习模型，包括各种图神经网络层和大量基准数据集。

如果我们暂时不了解某些概念，例如“ GCNConv”，请不要担心，之后我们将在以后的文章中介绍这些概念。

这个教程来自： `https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=ci-LpZWhRJoI` by [Matthias Fey](https://rusty1s.github.io/#/)

大家可以参照colab教程直接运行

### 4.1 PyG安装
- 查看当前torch和cuda版本
```
!python -c "import torch; print(torch.__version__)"
```
输出：`1.6.0`

```
!python -c "import torch; print(torch.version.cuda)"
```
输出：`10.1`


- 安装gpu版本
```
# 安装GPU版本
!pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install torch-geometric
```
可能安装gpu版本比较麻烦，大家可以多参照下官网进行安装：
`https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html`

![](https://files.mdnice.com/user/8341/0382bde5-e0ea-4a24-a2c3-0e5672675aba.png)

- 安装cuda版本
```
!pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-geometric
```
### 4.2 PyG数据集

PyTorch Geometric 可以通过 [`torch_geometric.datasets`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets) 快速的获取默认的数据集:

`https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html`


![](https://cdn.kesci.com/upload/image/qnfk9dmvwp.png?imageView2/0/w/960/h/960)

一共有20+种数据，比较丰富~

```
# 导入空手道俱乐部数据集
from torch_geometric.datasets import KarateClub
dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
```

输出：
```text
Dataset: KarateClub():
======================
Number of graphs: 1
Number of features: 34
Number of classes: 4
```

初始化[`KarateClub`]（https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.KarateClub） 数据集之后，我们首先可以检查其某些属性。
例如，我们可以看到该数据集恰好有**一个Graph**，并且该数据集中的每个节点都被分配了** 34维特征向量**（唯一地描述了空手道俱乐部的成员）。
此外，该图正好包含** 4个类别**，代表每个节点所属的社区。

现在让我们更详细地看一下这个Graph：
```
data = dataset[0]  # 获取graph对象.

print(data)
print('==============================================================')

# 获取一些graph的统计信息.
print(f'Number of nodes: {data.num_nodes}') # 节点的个数
print(f'Number of edges: {data.num_edges}') # 边的个属于
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 节点的度平均数
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```

输出如下：
```

Data(edge_index=[2, 156], train_mask=[34], x=[34, 34], y=[34])
==============================================================
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Number of training nodes: 4
Training node label rate: 0.12
Contains isolated nodes: False
Contains self-loops: False
Is undirected: True
```
通过上面的信息我们可以发现，这个KarateClub网络图一共有34个节点，边的个数为1，每个节点度的平均数有4.59，是一个无向图等等。

### 4.3 Data对象

PyTorch Geometric中的每个图形都由单个[`Data`]（https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data）对象表示，该对象包含所有 描述其图形表示的信息。
我们可以随时通过print（data）打印数据对象，以接收有关其属性及其形状的简短介绍：

```
Data(edge_index=[2, 156], x=[34, 34], y=[34], train_mask=[34])
```

我们可以看到这个`data`对象拥有4个属性：
（1）“ edge_index”属性保存有关“图形连接性”（即*）的信息，即每个边缘的源节点和目标节点索引的元组。
PyG进一步将
（2）节点特征**称为x（向34个节点中的每个节点分配了34维度的特征向量），将
（3）节点标记**称为y（每个节点仅分配给一个类别）。
（4）还有一个名为“ train_mask”的附加属性，它描述了我们已经知道其社区归属的节点。
总体而言，我们只知道4个节点的基本标签（每个社区一个），任务是推断其余节点的社区分配。

数据对象还提供了一些“实用功能”来推断基础图的一些基本属性。
例如，我们可以轻松推断图中是否存在孤立的节点（* ie *任何节点都没有边），图是否包含自环(*i.e.*, $(v, v) \in \mathcal{E}$)，或者图形是否是无向的（ (*i.e.*, for each edge $(v, w) \in \mathcal{E}$ there also exists the edge $(w, v) \in \mathcal{E}$).

```
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

edge_index = data.edge_index
# 打印节点
print(edge_index.t())
```
输出：
```
tensor([[ 0,  1],
        [ 0,  2],
        [ 0,  3],
        [ 0,  4],
        [ 0,  5],
        [ 0,  6],
        [ 0,  7],
        [ 0,  8],
        [ 0, 10],
        ....
        [33, 31],
        [33, 32]])
```

通过打印“ edge_index”，我们可以进一步了解PyG如何在内部表示图形连接。
我们可以看到，对于每个边缘，“ edge_index”都包含两个节点索引的元组，其中第一个值描述源节点的节点索引，第二个值描述边缘目标节点的节点索引。

这种表示称为** COO格式（坐标格式）**，通常用于表示稀疏矩阵。
代替以密集表示形式保存邻接信息 $\mathbf{A} \in \{ 0, 1 \}^{|\mathcal{V}| \times |\mathcal{V}|}$,，PyG稀疏地表示图形，这是指仅保存 $\mathbf{A}$ 中的条目为非零的坐标/值。

最后，我们可以通过将图形转换为`networkx`库格式来进一步可视化图形，该格式除了图形操作功能之外，还实现了强大的可视化工具。我们创建一个专门用于展示Graph的函数
```
%matplotlib inline
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()
```

```
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)
```

![](https://files.mdnice.com/user/8341/7471deed-b55b-4160-93a1-7960f15edec4.png)

### 4.4 实战：基于GCN的Graph节点分类

>接下来将通过Pytorch实现一个最基本的GCN网络用语节点分类，基于带有标签的节点数据进行训练模型，然后预测未带有标签的数据

在了解了PyG的数据处理之后，是时候实现我们的第一个Graph神经网络了！我们在这里将使用最简单的GNN网络之一，即GCN层**（[Kipf等人（2017）]（https://arxiv.org/abs/1609.02907））用于Graph节点分类。

PyG通过[`GCNConv`]（https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv）来实现此层，可以通过传入 节点要素表示形式“ x”和COO图形连接性表示形式“ edge_index”。

这样，我们就可以通过在torch.nn.Module类中定义我们的网络架构来创建我们的第一个图形神经网络：
```
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)
```
网络结构输出如下：
```
GCN(
  (conv1): GCNConv(34, 4)
  (conv2): GCNConv(4, 4)
  (conv3): GCNConv(4, 2)
  (classifier): Linear(in_features=2, out_features=4, bias=True)
)
```

在这里，我们首先在`__init__`中初始化所有构建块，并在“转发”中定义网络的计算流程。
我们首先定义并堆叠**三层图卷积层**，这对应于在每个节点周围（距离3个“跳”为止的所有节点）汇总3跳邻域信息。
另外，`GCNConv`层将节点特征维数减小为$2$, *i.e.*, $34 \rightarrow 4 \rightarrow 4 \rightarrow 2$.。每个[GCNConv]层都通过[tanh]（https://pytorch.org/docs/stable/generation/torch.nn.Tanh.html?highlight=tanh#torch.nn.Tanh）非线性进行了增强。

之后，我们应用单个线性变换（[`torch.nn.Linear`]（https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn。线性）），用作将我们的节点映射到4个类/社区中的1个的分类器。

我们返回最终分类器的输出以及GNN生成的最终节点嵌入。
我们继续通过`GCN()`初始化最终模型，并打印我们的模型以产生所有使用过的子模块的概括。

获取隐藏层的表示：
```
model = GCN()

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize(h, color=data.y)
```

输出：
```
Embedding shape: [34, 2]
```
![](https://files.mdnice.com/user/8341/7bef5fba-6d0d-450a-83f0-d618dcc2ad67.png)

在这里值得注意的是，即使在训练我们的模型权重之前，这个初始化的GCN网络也会产生节点的嵌入，并且这些嵌入与图的社区结构非常相似。尽管我们的模型的权重是“完全随机地初始化”的，并且到目前为止，我们还没有进行任何训练，但是相同颜色（社区）的节点已经在嵌入空间中紧密地聚集在一起。得出这样的结论，`即GNN引入了强烈的节点偏差，从而导致在输入图中彼此靠近的节点具有相似的嵌入`。



但是，节点表示并不是很完美，我们可以看出来四种类别的节点还是混在一块了（对应途中是四种颜色），我们可以做得更好吗？
让我们看一个示例，该示例如何基于对图中4四种节点的社区分配（每个社区一个）的知识来训练我们的网络参数：

由于模型中的所有内容都是可区分的和参数化的，因此我们可以添加一些标签，训练模型并观察嵌入如何反应。
在这里，我们使用一种半监督学习程序：我们仅针对每个类训练一个节点，但是允许使用完整的输入图数据。

这个怎么理解呢，我们其实可以输出`data.y`和`data.train_mask`就明白了:
```
data.y
tensor([0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2,
        3, 3, 2, 2, 3, 2, 2, 3, 2, 2])
```
```
data.train_mask
tensor([ True, False, False, False,  True, False, False, False,  True, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False,  True, False, False, False, False, False,
        False, False, False, False])

```
大家可以看到相当于我们训练网络的时候只针对每一个类别下的token做优化，这样可以加速网络的训练和收敛。

训练我们的模型与任何其他PyTorch模型非常相似。
除了定义我们的网络架构之外，我们还定义了损失函数在这里[[CrossEntropyLoss]]（https://pytorch.org/docs/stable/generate/torch.nn.CrossEntropyLoss.html）） 并初始化随机梯度优化器（此处为[`Adam`]（https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam））。
之后，我们执行多轮优化，其中每一轮都包含一个正向和反向传递，以计算模型参数w.r.t的梯度。从前向通行证产生的损失。
如果您对PyTorch并不陌生，则该方案应该对您来说很熟悉。
否则，PyTorch文档会提供[有关如何在PyTorch中训练神经网络的很好的介绍]（https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-loss-function-and-optimizer ）。

请注意，我们的半监督学习场景是通过以下行实现的：
```
loss = criterion(out[data.train_mask], data.y[data.train_mask])
```
在计算所有节点的节点嵌入时，我们“仅利用训练节点来计算loss” **。
在这里，这是通过过滤分类器“ out”和真实性标签“ data.y”的输出以仅包含“ train_mask”中的节点来实现的。

现在让我们开始训练，看看我们的节点嵌入随时间如何演变（最好是显式地运行代码，打印出模型的训练效果）：

```
import time
from IPython.display import Javascript  #画图.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 430})'''))

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # 定义loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

def train(data):
    optimizer.zero_grad()  # 梯度清零.
    out, h = model(data.x, data.edge_index)  # GCN模型.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算真实loss.
    loss.backward()  # 反向求导.
    optimizer.step()  # 参数更新.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    # 每10个epochs打印Graph
    if epoch % 10 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)
```


![](https://files.mdnice.com/user/8341/d4bdba4b-2f5d-48df-8919-e0e2feb16a76.png)

![](https://files.mdnice.com/user/8341/3fccbd1a-c04a-427a-aad0-bd66bf1cc0da.png)

过了一段时间之后，我们可以看到GCN强大的效果：


![](https://files.mdnice.com/user/8341/b76e4545-eda0-48ee-9e65-9226700553a9.png)


![](https://files.mdnice.com/user/8341/8faf8997-1b06-4fa3-8995-03abc5a5f7d0.png)


可以看到，我们的3层GCN模型可以有效地社区发现并正确地对大多数节点进行分类。
