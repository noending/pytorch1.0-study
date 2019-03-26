### **Pytorch-1.0 Cookbook**
#### 1.0 常用包
```
import collections
import os

import numpy as np
import PIL.Image
import torch
import torchvision

import torch.nn as nn
import torchvison.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils， datasets
```

#### 2.1.0 数据预处理及导入

```
# Define a transform to normalize the data 归一化数据预处理
transform = transforms.Compose([transforms.RandomSizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5),(0.5,)),
                               ])

```
具体说明，可以查看[torchvision.transforms文档](https://pytorch.org/docs/stable/torchvision/transforms.html)

##### 2.1.1 从torchvison包里下载数据集

```
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```
MNIST可以替换为Fashion-MNIST，CIFAR, VOC,COCO等，可下载使用的[数据集](https://pytorch.org/docs/stable/torchvision/datasets.html)

##### 2.1.2 从本地文件夹中访问数据集

```
data_dir = 'Cat_Dog_data/train'

dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

```
[具体ImageFolder,DatasetFolder说明](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)

***

