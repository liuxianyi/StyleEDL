import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import os


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1), nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x, adj):
        x = torch.matmul(adj, x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        # import pdb; pdb.set_trace()
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.softmax(dynamic_adj, dim=-1) # A_d
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x, adj):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x, adj) # x [8, 3920, 8]
        x = x + out_static  # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x

class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(HighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(
                    self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                setattr(
                    self, name,
                    nn.Sequential(
                        nn.Conv2d(in_channels,
                                  self.inter_channels,
                                  1,
                                  padding=0,
                                  bias=False)))
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i + 1)
            setattr(
                self, name,
                nn.Sequential(
                    nn.Conv2d(self.inter_channels,
                              in_channels,
                              1,
                              padding=0,
                              bias=False), nn.Sigmoid()))

    def forward(self, x):
        y = []
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(
                    self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt = 0
        for j in range(self.order):
            y_temp = 1
            for i in range(j + 1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))


        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i + 1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out 


class GMG(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.parts = opt["parts"] 
        self.class_num = opt["num_classes"]
        self.lambda_ = opt['lambda']
        self.resnet50 = models.resnet50(pretrained=True)


        self.g1_sample = nn.UpsamplingNearest2d(size=(224, 224))
        self.g2_sample = nn.UpsamplingNearest2d(size=(224, 224))
        self.g3_sample = nn.UpsamplingNearest2d(size=(224, 224))

        self.x_up = nn.UpsamplingNearest2d(size=(28, 28))

        self.gram_ln = nn.LayerNorm((3, 224, 224))
        self.gram_conv1 = nn.Conv2d(in_channels=3,
                                    out_channels=8,
                                    kernel_size=7,
                                    padding=3)
        self.gram_conv1_ln = nn.LayerNorm(224)
        self.gram_relu1 = nn.ReLU()
        self.gram_pool1 = nn.MaxPool2d(2)

        self.gram_conv2 = nn.Conv2d(in_channels=8,
                                    out_channels=16,
                                    kernel_size=7,
                                    padding=3)
        self.gram_conv2_ln = nn.LayerNorm(112)
        self.gram_relu2 = nn.ReLU()
        self.gram_pool2 = nn.MaxPool2d(2)

        self.x_conv = nn.Conv2d(2048, 256, 1)
        self.x_relu = nn.ReLU()

        self.l2_conv = nn.Conv2d(1024, 256, 1)
        self.l2_relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Conv2d(512, 8, 1)
        # MHN
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            setattr(self, name, HighDivModule(512, i + 1)) # High order feature_map channel attention, 
            # h*w spatial location point 
            # paper: https://ieeexplore.ieee.org/document/9009039/
            # link: https://zhuanlan.zhihu.com/p/104380548

        for i in range(self.parts):
            name = 'classifier' + str(i)
            setattr(self, name,
                    nn.Sequential(nn.Conv2d(512, self.class_num, 1)))

        for i in range(self.parts):
            name = 'classifier2' + str(i)
            setattr(self, name,
                    nn.Sequential(nn.Conv2d(320, self.class_num, 1)))

        # GCN
        self.adj = torch.from_numpy(
            np.load(
                os.path.join(opt["data_path"], opt['dataset'], "twitter.npy") )
        ).float().to(opt['device'])

        self.gcn = DynamicGraphConvolution(980*self.parts, 980*self.parts, 8) # style-gcn
    
    @staticmethod
    def _cal_gram(feature):
        feature = feature.view(feature.shape[0], feature.shape[1], -1)
        feature = torch.matmul(feature, feature.transpose(-1, -2))
        return feature

    def _gram_forward(self, g1, g2, g3):
        g1 = self._cal_gram(g1).unsqueeze(1) # b * c1 * wh -> b * 1 * c1 * c1
        g2 = self._cal_gram(g2).unsqueeze(1) # b * c2 * wh -> b * 1 * c2 * c2
        g3 = self._cal_gram(g3).unsqueeze(1) # b * c2 * wh -> b * 1 * c3 * c3
        g1 = self.g1_sample(g1) # b * 1 * 224 * 224
        g2 = self.g2_sample(g2) # b * 1 * 224 * 224
        g3 = self.g3_sample(g3) # b * 1 * 224 * 224
        g = torch.cat([g1, g2, g3], dim=1) # b * 3 * 224 * 224

        g = self.gram_ln(g)
        g = self.gram_conv1(g)
        g = self.gram_relu1(g)
        g = self.gram_conv1_ln(g)
        g = self.gram_pool1(g) # b * 8 * 112 * 112
        g = self.gram_conv2(g)
        g = self.gram_relu2(g)
        g = self.gram_conv2_ln(g)
        g = self.gram_pool2(g) # b * 16 * 56 * 56

        g1 = g.view(g.shape[0], -1, 14, 14) # b * (16*4*4) * 14 * 14
        g1 = g1.repeat(self.parts, 1, 1, 1) # b parts * 16 * 14 * 14

        g2 = g.view(g.shape[0], -1, 28, 28) # b * (16*2*2) * 28 * 28
        g2 = g2.repeat(self.parts, 1, 1, 1) # b parts * (16*2*2) * 28 * 28
        return g1, g2

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        g1 = x
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        g2 = x
        x = self.resnet50.layer2(x)
        g3 = x # b, c, h, w

        # High-Order Attention
        xx = []
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            layer = getattr(self, name)
            xx.append(layer(x))


        x = torch.cat(xx, 0) # B*parts, 512, 56, 56

        x = self.resnet50.layer3(x) # append high order attention on feature map 
        # with the output of layer2
        # B*parts, 1024, 28, 28

        #  conv of 1*1 with relu activate behind of the output of layers for getting fc2, l2
        l2 = self.l2_conv(x)  # (B*parts)*256*28*28
        fc2 = l2 # -> 
        l2 = self.l2_relu(l2)

        x = self.resnet50.layer4(x)  # (B*parts)*2048*14*14

    
        x = self.x_conv(x) # (B*parts)*256*14*14
        fc1 = x # ->
        x = self.x_relu(x)
        xxxx = x

        # Style Representation
        g1, g2 = self._gram_forward(g1, g2, g3)# two scale style feature
        # Input: (B, 64, 224, 224), (B, 256, 112, 112), (B, 512, 56, 56)
        # Input: (B*parts, 256, 14, 14), (B*parts, 64, 28, 28)

        xx1 = torch.cat([x, g1], dim=1) # (B*parts)*512*14*14
        num = int(xx1.size(0) / self.parts) # batch

        y_1 = []
        for i in range(self.parts):
            name = 'classifier' + str(i)
            layer = getattr(self, name)
            x = layer(xx1[i * num:(i + 1) * num, :])
            y_1.append(x.flatten(2))
        # y_1: list (B, 8, 196) len=2

        # FPN two layer
        xxxx = self.x_up(xxxx) # (B*parts)*256*28*28
        l2 = xxxx + l2
        xx2 = torch.cat([l2, g2], dim=1) # # (B*parts)*(256+64)*28*28

        y_2 = []
        for i in range(self.parts):
            name = 'classifier2' + str(i)
            layer = getattr(self, name)
            x = layer(xx2[i * num:(i + 1) * num, :])
            y_2.append(x.flatten(2))
        # y_2: list (B, 8, 784) len=parts

        # cat
        # 1. cat fpn 
        y_ = []
        for i in range(len(y_1)):
            y_.append(torch.cat([y_1[i], y_2[i]], dim=-1))
        # y_: list (B, 8, 980) len=parts

        # 2. mhn
        # for GCN
        yy_ = y_[0]
        for i in range(1, len(y_)):
            yy_ = torch.cat((yy_, y_[i]), dim=-1)
        # yy_: (B, 8, 1960) len=parts

        # mixed attention
        for i in range(len(y_)):
            base_logit = torch.mean(y_[i], dim=2) # first Order (B, 8)
            att_logit = torch.max(y_[i], dim=2)[0] # second Order (B, 8)
            y_[i] = self.softmax(base_logit + self.lambda_ * att_logit)

        # GCN
        yy_ = yy_.transpose(1, 2) # (B, 1960, 8)
        yy_ = self.gcn(yy_, self.adj) # adj 8, 8
        yy = yy_.transpose(1, 2) # (B, 8, 1960)
        base_logit = torch.mean(yy, dim=2) # (B, 8)
        att_logit = torch.max(yy, dim=2)[0] # (B, 8)
        gcn = self.softmax(base_logit + self.lambda_ * att_logit)

        return y_, gcn, fc1, fc2
