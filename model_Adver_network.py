# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):          #这是一个继承自 torch.autograd.Function 的自定义自动求导函数,主要作用是在反向传播时,
                                        # 将梯度乘以一个系数 alpha,从而实现对抗训练中的梯度反转层(Gradient Reversal Layer)
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha               #在前向传播时,这个函数直接返回输入 x,不做任何操作。
        return x.view_as(x)             #但是会将 alpha 的值保存在 ctx.alpha 中,以便在反向传播时使用。

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha    #在反向传播时,这个函数将输入的梯度 grad_output 取反,然后乘以保存的 alpha 值。最后返回乘积作为输出的梯度。
        return output, None                       #第二个返回值 None 表示这个函数没有对 alpha 进行求导。


class Discriminator(nn.Module):                                                 #这是一个判别器模型,用于对抗训练中的域分类任务。
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=10000):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim                                #input_dim: 输入特征的维度,默认为256。
        self.hidden_dim = hidden_dim                              #hidden_dim: 隐藏层的维度,默认为256。

        layers = [                                                #num_domains: 域的数量,默认为4。
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),                                            #一个简单的全连接神经网络,包含两个隐藏层
            nn.Linear(hidden_dim, hidden_dim),                    #每个隐藏层后接一个批归一化层和ReLU激活函数
            nn.BatchNorm1d(hidden_dim),                           #最后一层的输出维度为 num_domains,对应域的数量。
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]


        self.layers = torch.nn.Sequential(*layers)                #在前向传播时,输入特征 x 直接通过网络的各层,得到域分数的输出。

    def forward(self, x):
        return self.layers(x)


