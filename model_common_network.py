
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):                                             #这个模块主要用于特征降维。
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)                #这一行定义了一个批归一化层，用于对输入特征进行归一化处理，提高训练稳定性。bottleneck_dim是输入特征的维度。
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)             #这一行定义了一个全连接层，用于将输入的feature_dim维特征映射到bottleneck_dim维空间。
        self.type = type


    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        x = self.dropout(x)
        return x


class feat_classifier(nn.Module):                                            #这个模块是一个分类器，用于基于特征输出对应的类别分数
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
