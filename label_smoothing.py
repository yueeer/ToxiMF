import torch
import torch.nn as nn
from torch.autograd import Variable


'''
reference : https://cloud.tencent.com/developer/article/1684298
'''

class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean', label_num=2):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
        self.label_num = label_num

    def _one_hot(self, labels, value):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        result = torch.zeros(labels.size(0), self.label_num)
        true_label = torch.Tensor(labels.size(0)).fill_(1 - value).to(labels.device)
        result[:,1] = labels * true_label
        false_label = torch.Tensor(labels.size(0)).fill_(value / (self.label_num - 1)).to(labels.device)
        result[:,0] = false_label * (-1 * labels + 1)
        return result

    def _smooth_label(self, target, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, value=smooth_factor)
        return one_hot.to(target.device)

    def forward(self, x, e):
        smoothed_target = self._smooth_label(x, e)
        return smoothed_target


# -*- coding: utf-8 -*-

"""
qi=1-smoothing(if i=y)
qi=smoothing / (self.size - 1) (otherwise)#所以默认可以fill这个数，只在i=y的地方执行1-smoothing
另外KLDivLoss和crossentroy的不同是前者有一个常数
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],

                                 [0, 0.9, 0.2, 0.1, 0], 

                                 [1, 0.2, 0.7, 0.1, 0]])
对应的label为
tensor([[ 0.0250,  0.0250,  0.9000,  0.0250,  0.0250],
        [ 0.9000,  0.0250,  0.0250,  0.0250,  0.0250],
        [ 0.0250,  0.0250,  0.0250,  0.9000,  0.0250]])
区别于one-hot的
tensor([[ 0.,  0.,  1.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.]])
"""


class LabelSmoothing(nn.Module):
    "Implement label smoothing.  size表示类别总数  "

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False)

        # self.padding_idx = padding_idx

        self.confidence = 1.0 - smoothing  # if i=y的公式

        self.smoothing = smoothing

        self.size = size

        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        # print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))


