import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random


from EMSA import EMSA
from ExternalAttention import  ExternalAttention
from AFT import AFT_FULL


from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv,GATConv


class GATNet(torch.nn.Module):
    def __init__(self, num_feature, out_feature,him):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, him, heads=8, concat=True, dropout=0.6)
        self.GAT2 = GATConv(8*him, out_feature, dropout=0.6)

    def forward(self, x,edge_index):
        # x, edge_index = data.x, data.edge_index

        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.GAT2(x, edge_index)
        return x

class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1,
                 hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True,
                 feature_fusion=None, class_num=7):
        super(GIN_Net2, self).__init__()
        # self.emsa = ExternalAttention(d_model=2000,S=13)
        self.alt_full = AFT_FULL(d_model=2000, n=13)
        self.gat =GATNet(256,512,10)
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion

        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size), gin_in_feature)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)

    def reset_parameters(self):

        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()

        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()

    def forward(self, x, edge_index, train_edge_id, p=0.5):
        # y = torch.randn(5189, 2, 2000, 2000)
        # y =
        x = x.transpose(1, 2)
        # print(x.shape)
        x= self.alt_full(x)
        # print(x.shape)
        x = self.conv1d(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.maxpool1d(x)
        # print(x.shape)


        #
        # x = x.transpose(1, 2)
        # # print(x.shape)
        # x, _ = self.biGRU(x)
        # # print(x.shape)
        # x = self.global_avgpool1d(x)
        # # print(x.shape)



        x = x.squeeze()
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        # print(x.shape)
        # print(edge_index.shape)
        x=self.gat(x, edge_index)
        #
        # print(x.shape)
        # print(x.shape)
        # x = self.gin_conv1(x, edge_index)
        # xs = [x]
        # for conv in self.gin_convs:
        #     x = conv(x, edge_index)
        #     xs += [x]
        #
        # if self.use_jk:
        #     x = self.jump(xs)

        x = F.relu(self.lin1(x))
        # print(x.shape)
        x = F.dropout(x, p=p, training=self.training)
        # print(x.shape)
        x = self.lin2(x)
        # print(x.shape)
        # x  = torch.add(x, x_)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        # print("x1",x1.shape)
        x2 = x[node_id[1]]
        # print("x2",x2.shape)

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)

        return x