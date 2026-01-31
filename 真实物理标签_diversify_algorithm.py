# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from model_diversify_network import CWRUNetwork

import model_Adver_network, model_common_network
from model_base import Algorithm
from model_common_loss import Entropylogits
from config import Config

# 使用 Config 类实例化配置
config = Config()


class Diversify(Algorithm):

    def __init__(self, config):

        super(Diversify, self).__init__(config)
        self.first_round = True  # 添加初始化标志
        # 将所有模型移到指定设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.featurizer = CWRUNetwork()

        # 第一步,获得细粒度特征
        self.abottleneck = model_common_network.feat_bottleneck(self.featurizer.in_features, config.bottleneck, config.layer)
        self.aclassifier = model_common_network.feat_classifier(config.num_classes * config.latent_domain_num, config.bottleneck, config.classifier)
        self.discriminator = model_Adver_network.Discriminator(config.bottleneck, config.dis_hidden, config.latent_domain_num)

        # 用于分类任务,
        self.bottleneck = model_common_network.feat_bottleneck(self.featurizer.in_features, config.bottleneck, config.layer)
        self.classifier = model_common_network.feat_classifier(config.num_classes, config.bottleneck, config.classifier)

        # 用于域适应任务
        self.dbottleneck = model_common_network.feat_bottleneck(self.featurizer.in_features, config.bottleneck, config.layer)
        self.ddiscriminator = model_Adver_network.Discriminator(config.bottleneck, config.dis_hidden, config.num_classes)
        self.dclassifier = model_common_network.feat_classifier(config.latent_domain_num, config.bottleneck, config.classifier)

        # 用于 t-SNE 记录数据
        self.tsne_features_before_d = []  # 记录 update_d 之前的特征
        self.tsne_domain_labels_before_d = []  # update_d 之前的潜在领域标签
        self.tsne_class_labels_before_d = []  # 记录 update_d 之前的类别标签

        self.tsne_features_d = []  # 记录 update_d 之后的特征
        self.tsne_domain_labels_d = []  # update_d 之后的潜在领域标签
        self.tsne_class_labels_d = []  # 记录 update_d 之后的类别标签

        self.tsne_features = []  # 记录 update 之后的特征
        self.tsne_domain_labels = []  # update 之后的潜在领域标签
        self.tsne_class_labels = []  # 记录 update 之后的类别标签

        self.is_last_epoch = False  # 标记是否是最后一个 epoch
        self.is_first_epoch = True  # 记录是否是第一轮

        self.tsne_colors = []  # 记录颜色映射信息
        self.tsne_markers = []  # 记录形状信息


    def set_dlabel(self, loader):                                           #这个函数用于为数据集中的样本设置域标签

        self.dbottleneck.eval()
        self.dclassifier.eval()                                             #通常在做预测时,我们会将模型设置为评估模式。
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.cuda().float()                              #将输入数据inputs移动到GPU上,并转换为浮点数。
                index = data[-1]                                            #获取当前样本的索引index。
                feas = self.dbottleneck(self.featurizer(inputs))            #将输入数据inputs传递给self.featurizer获取特征表示,再传给self.dbottleneck获取瓶颈特征feas。
                outputs = self.dclassifier(feas)                            #将feas传给self.dclassifier获取域预测概率outputs

                if start_test:                                              #如果是第一次迭代,将feas和outputs保存到all_fea和all_output中
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:                                                        #否则,将它们与之前的特征和预测概率进行拼接。
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat( (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))                #将当前样本的索引index保存到all_index中

                                                                             #获得了所有样本的特征表示all_fea、域预测概率all_output和对应的索引all_index。
        all_output = nn.Softmax(dim=1)(all_output)                           #对all_output进行softmax操作,得到每个样本属于不同域的概率分布

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()        #对all_fea进行归一化处理,方法是在特征矩阵的最后一列添加1,再对每一行进行L2范数归一化。
        all_fea = all_fea.float().cpu().numpy()                              #将all_fea转换为NumPy数组。

        K = all_output.size(1)                                               #获取域的数量K。
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)                                 #计算域概率分布aff的转置与特征矩阵all_fea的乘积,作为初始域中心initc。
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])                    #对initc进行归一化处理。
        dd = cdist(all_fea, initc, 'cosine')                           #计算每个样本与所有域中心之间的余弦距离dd。
        pred_label = dd.argmin(axis=1)                                       #将每个样本分配到距离最近的域中心,得到初始域标签pred_label

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)                             #基于pred_label,重新计算域中心initc
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')                       #再次计算样本与新域中心的距离,重新分配域标签pred_label
            pred_label = dd.argmin(axis=1)                                   #获得了每个样本的域标签pred_label

        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel') #将计算出的域标签pred_label保存到数据集中
        print(Counter(pred_label))                                           #打印每个域的样本数量
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

 # c) 联合更新(update_a)：
    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float()               # 输入数据
        all_c = minibatches[1].cuda().long()                # 类别标签
        all_d = minibatches[4].cuda().long()                # 领域标签

        all_y = all_d * config.num_classes + all_c                  # 使用 config.num_classes 组合领域标签和类别标签
        all_z = self.abottleneck(self.featurizer(all_x))            # 将输入数据all_x传递给特征提取网络self.featurizer获取特征表示,再通过self.abottleneck层获取瓶颈特征all_z
        all_preds = self.aclassifier(all_z)                         # 将all_z输入到联合分类器self.aclassifier中,获取预测输出all_preds。

        classifier_loss = F.cross_entropy(all_preds, all_y)         # 计算all_preds对应于联合标签all_y的交叉熵损失classifier_loss。
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}

    def record_initial_features(self, train_loader):
        """ 在 update_d 之前，记录初始特征 """
        print("📢 记录 update_d 之前的初始特征...")
        self.tsne_features_before_d.clear()
        self.tsne_domain_labels_before_d.clear()
        self.tsne_class_labels_before_d.clear()  # 记录类别标签
        self.tsne_colors.clear()
        self.tsne_markers.clear()

        with torch.no_grad():
            for minibatch in train_loader:
                all_x1 = minibatch[0].to(self.device).float()
                all_c1 = minibatch[1].to(self.device).long()
                all_d1 = minibatch[4].to(self.device).long()  # 初始领域标签 (默认是 0)

                z1 = self.dbottleneck(self.featurizer(all_x1))
                self.tsne_features_before_d.append(z1.cpu().numpy())
                self.tsne_domain_labels_before_d.append(all_d1.cpu().numpy())
                self.tsne_class_labels_before_d.append(all_c1.cpu().numpy())  # 形状

#a) 域判别器更新(update_d)：
    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].cuda().float()            # 输入特征
        all_d1 = minibatch[1].cuda().long()             # 域标签
        all_c1 = minibatch[4].cuda().long()             # 类别标签

        z1 = self.dbottleneck(self.featurizer(all_x1))                        #计算特征提取网络的输出 z1

        disc_in1 = model_Adver_network.ReverseLayerF.apply(z1, config.alpha1)    #通过反向层 ReverseLayerF 进行梯度反转,得到 disc_in1
        disc_out1 = self.ddiscriminator(disc_in1)                             #使用域判别器 self.ddiscriminator 对输入的域标签 all_d1 进行预测
        disc_loss = F.cross_entropy(disc_out1, all_d1)                          #计算交叉熵损失 disc_loss

        cd1 = self.dclassifier(z1)                                            #将 z1 输入到分类器 self.dclassifier 中,获得输出 cd1
        ent_loss = Entropylogits(cd1)*config.lam + F.cross_entropy(cd1, all_c1)

        loss = ent_loss+0.1*disc_loss                                             #将两个损失相加作为总损失
        opt.zero_grad()                                                       #并使用优化器 opt 进行反向传播和参数更新。
        loss.backward()
        opt.step()

        # 仅在最后一个 epoch 记录潜在领域数据
        if self.is_last_epoch:
            self.tsne_features_d.append(z1.detach().cpu().numpy())                      # 记录特征
            self.tsne_domain_labels_d.append(all_c1.detach().cpu().numpy())             # 记录潜在领域标签
            self.tsne_class_labels_d.append(all_d1.detach().cpu().numpy())              # 形状

        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

#b) 分类器更新(update)：
    def update(self, data, opt):
        all_x = data[0].cuda().float()
        all_y = data[1].cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))                #将all_x输入到特征提取网络self.featurizer中获取特征表示,再通过self.bottleneck层获取瓶颈特征all_z。

        disc_input = all_z                                             #将瓶颈特征all_z作为输入,通过Adver_network.ReverseLayerF层进行梯度反转,得到disc_input。
        all_preds = self.classifier(all_z)                             #将瓶颈特征all_z输入到分类器self.classifier中,获取类别预测输出all_preds
        classifier_loss = F.cross_entropy(all_preds, all_y)            #计算类别预测的交叉熵损失classifier_loss。


        disc_input = model_Adver_network.ReverseLayerF.apply(disc_input, config.alpha)
        disc_out = self.discriminator(disc_input)                      #将disc_input输入到域判别器self.discriminator中,获取域预测输出disc_out
        disc_labels = data[2].cuda().long()                            #从输入的data中获取域标签disc_labels,并将其移动到GPU上,转换为长张量
        disc_loss = F.cross_entropy(disc_out, disc_labels)             #计算域预测的交叉熵损失disc_loss。


        loss = classifier_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 仅记录最后一个 epoch 数据
        if self.is_last_epoch:
            self.tsne_features.append(all_z.detach().cpu().numpy())
            self.tsne_domain_labels.append(disc_labels.detach().cpu().numpy())
            self.tsne_class_labels.append(all_y.detach().cpu().numpy())  # 形状


        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}



    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))


