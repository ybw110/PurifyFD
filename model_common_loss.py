# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F


def Entropy(input_):                                    #这个函数计算输入张量 input_ 的熵。
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)     #计算每个元素的熵。由于对数函数在 0 处是无定义的,因此添加一个很小的值 epsilon 以避免计算溢出。
    entropy = torch.mean(torch.sum(entropy, dim=1))     #首先对每个样本的熵求和,得到每个样本的总熵;然后计算所有样本总熵的平均值,作为最终输出。
    return entropy


def Entropylogits(input, redu='mean'):
    input_ = F.softmax(input, dim=1)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if redu == 'mean':
        entropy = torch.mean(torch.sum(entropy, dim=1))  #如果 redu='mean',则计算所有样本的平均熵,与 Entropy 函数的输出相同。
    elif redu == 'None':
        entropy = torch.sum(entropy, dim=1)              #如果 redu='None',则返回每个样本的总熵,不做平均操作。
    return entropy


                                #这两个函数提供了计算张量熵的功能,可以用于监控模型输出的不确定性,或作为正则化项加入损失函数中,以提高模型的泛化能力。
                                # Entropy 函数直接计算输入张量的熵,
                                # 而 Entropylogits 函数则先对输入进行 softmax 操作,再计算softmax输出的熵,提供了更多的灵活性。