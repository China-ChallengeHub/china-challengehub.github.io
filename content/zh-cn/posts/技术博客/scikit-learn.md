+++
title = "Scikit-Learn"
date = "2021-03-27"
description = "关于 Scikit-Learn 的详细描述"
featured = false
categories = [
  "技术博客"
]
tags = [
  "scikit-learn"
]
series = [
  "Scikit-Learn系列教程"
]
images = [
]
+++

本文展示了如果使用 `scikit-learn` 使用。
<!--more-->
sklearn (scikit-learn) 是基于 Python 语言的机器学习工具

> 官方链接：https://scikit-learn.org/stable/



## 安装
- 通过pip安装
```shell
$ pip install -U scikit-learn
```
- 通过conda安装
```shell
$ conda install -c intel scikit-learn
```

## 实例

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
```
```text
RandomForestClassifier(random_state=0)
```
