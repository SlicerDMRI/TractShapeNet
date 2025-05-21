import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PointNet(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3, task_type='reg'):  # Added num_outputs parameter
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        if task_type=='reg':
            self.fc3 = nn.Linear(256, 5) #Change Change Change
        if task_type == 'cla':
            self.fc3 = nn.Linear(256, num_outputs)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(512)
        self.bnf2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.task_type=task_type
    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.size())
        x = x.max(dim=-1, keepdim=False)[0]
        #print(x.size())
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.relu(self.bnf2(self.fc2(x)))
        #x = F.relu(self.bnf2(self.dropout(self.fc2(x))))
        if self.task_type=='reg':
            score = self.fc3(x)
        elif self.task_type=='cla':
            x=self.fc3(x)
            score=F.log_softmax(x,dim=1)
        return score

class PointNet_vis(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,task_type='reg'):
        super(PointNet_vis, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        if task_type=='reg':
            self.fc3 = nn.Linear(256, 1)
        if task_type == 'cla':
            self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(512)
        self.bnf2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.task_type=task_type
        # self.fc31 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        # self.fc32 = nn.Linear(256, 1)
    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.size())
        #xs=x.sum(dim=-1, keepdim=False)
        x1,ids = x.max(dim=-1, keepdim=False)
        x1r=(x1.unsqueeze(2)).repeat(1,1,x.shape[2])
        xs=torch.sum(x1r==x,2)
        ids[xs!=1]=-1
        #print(x.size())
        x=x1
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.relu(self.bnf2(self.fc2(x)))
        #x = F.relu(self.bnf2(self.dropout(self.fc2(x))))
        if self.task_type=='reg':
            score = self.fc3(x)
        elif self.task_type=='cla':
            x=self.fc3(x)
            score=F.log_softmax(x,dim=1)
        return score,ids

class PointNet_feature(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,task_type='reg'):
        super(PointNet_feature, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.task_type=task_type
    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.size())
        x = x.max(dim=-1, keepdim=False)[0]
        #print(x.size())
        x = F.relu(self.bnf1(self.fc1(x)))
        feature=self.fc2(x)
        return feature
class PointNet_all(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,n_tract=20,task_type='reg'):
        super(PointNet_all, self).__init__()
        #self.pn=PointNet(input_channel=input_channel,task_type=task_type)
        for i in range(n_tract):
            exec("self.pn{} = PointNet(input_channel=input_channel,task_type=task_type)".format(i))
            exec("for param in self.pn{}.parameters(): param.requires_grad=False".format(i))
        self.fc = nn.Linear(n_tract, 1)
        # self.fc.weight.data.fill_(1/n_tract)
        # self.fc.bias.data.zero_()
        self.fc.weight.data.fill_(2.66222683/n_tract)
        self.fc.bias.data.fill_(-0.7681)
        self.n_tract=n_tract
    def forward(self, x):
        n_points=int(x.size()[2]/self.n_tract)
        score_tracts=None
        for i in range(self.n_tract):
            x_tract=x[:,:,n_points*i:n_points*(i+1)]
            score_tract=eval("self.pn{}(x_tract)".format(i))
            if score_tracts==None:
                score_tracts=score_tract
            else:
                score_tracts = torch.cat((score_tracts,score_tract),1)
        score=self.fc(score_tracts)
        #print(score)
        #print(self.pn0.conv1.bias[:5])
        return score
class PointNet_all_feature(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,n_tract=20,task_type='reg'):
        super(PointNet_all_feature, self).__init__()
        #self.pn=PointNet(input_channel=input_channel,task_type=task_type)
        for i in range(n_tract):
            exec("self.pn{} = PointNet_feature(input_channel=input_channel,task_type=task_type)".format(i))
            exec("for param in self.pn{}.parameters(): param.requires_grad=False".format(i))
        #self.conv1=nn.Conv1d(20, 32, 5, stride=3)
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, 3, stride=3)
        self.conv3 = nn.Conv1d(64, 128, 3, stride=3)
        #self.fc1 = nn.Linear(1152, 256)
        self.fc1 = nn.Linear(256*20, 256*4)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(256 * 4,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        # self.fc = nn.Linear(n_tract, 1)
        # self.fc.weight.data.fill_(2.66/n_tract)
        # #self.fc.bias.data.zero_()
        # self.fc.bias.data.fill_(-0.84)
        self.n_tract=n_tract
    def forward(self, x):
        n_points=int(x.size()[2]/self.n_tract)
        score_tracts=None
        for i in range(self.n_tract):
            x_tract=x[:,:,n_points*i:n_points*(i+1)]
            score_tract=eval("self.pn{}(x_tract)".format(i))
            #score_tract=score_tract.unsqueeze(1)
            if score_tracts==None:
                score_tracts=score_tract
            else:
                score_tracts = torch.cat((score_tracts,score_tract),1)
        #print(score_tracts.size())
        # x1 = F.relu(self.conv1(score_tracts))
        # x2 = F.relu(self.conv2(x1))
        # x3 = F.relu(self.conv3(x2))
        # x4 = x3.view(x.size(0), -1)
        # x5 = F.relu(self.fc1(x4))
        # score = self.fc2(x5)

        x = F.relu(self.bn1(self.fc1(score_tracts)))
        x = F.relu(self.bn2(self.fc2(x)))
        score = self.fc3(x)
        # x=torch.mean(score_tracts,1,keepdim=False)
        # score=self.fc2(x)
        #print(score)
        #print(self.pn0.conv1.bias[:5])
        return score
class PointNet_cl(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,task_type='reg'):
        super(PointNet_cl, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.fc31 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.fc32 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(512)
        self.bnf2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.task_type=task_type
    def forward(self, x1,x2):
        def forward_single(self, x):
            #print(x.size())
            x = F.relu(self.bn1(self.conv1(x)))
            #print(x.size())
            x = F.relu(self.bn2(self.conv2(x)))
            #print(x.size())
            x = F.relu(self.bn3(self.conv3(x)))
            #print(x.size())
            x = x.max(dim=-1, keepdim=False)[0]
            #print(x.size())
            x = F.relu(self.bnf1(self.fc1(x)))
            x = F.relu(self.bnf2(self.fc2(x)))
            score=self.fc3(x)
            #x = self.fc31(x)
            return x,score
        e1,score1=forward_single(self, x1)
        e2,score2 = forward_single(self, x2)
        dis=self.fc32(e1-e2)
        return dis,score1,score2
class PointNet_cla(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,task_type='reg'):
        super(PointNet_cla, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        if task_type=='reg':
            self.fc3 = nn.Linear(256, 1)
        if task_type == 'cla':
            self.fc3 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(512)
        self.bnf2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.task_type=task_type
    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.size())
        x = x.max(dim=-1, keepdim=False)[0]
        #print(x.size())
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.relu(self.bnf2(self.fc2(x)))
        #x = F.relu(self.bnf2(self.dropout(self.fc2(x))))
        if self.task_type=='reg':
            score = self.fc3(x)
        elif self.task_type=='cla':
            x=self.fc3(x)
            score=F.log_softmax(x,dim=1)
        return score
class PointNets(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,task_type='reg'):
        super(PointNets, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 256)
        if task_type=='reg':
            self.fc3 = nn.Linear(256, 1)
        if task_type == 'cla':
            self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.task_type=task_type
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        x = F.relu(self.bnf1(self.fc1(x)))
        if self.task_type=='reg':
            score = self.fc3(x)
        elif self.task_type=='cla':
            x=self.fc3(x)
            score=F.log_softmax(x,dim=1)
        return score

# class PointNet(nn.Module):
#     # from DGCNN's repo
#     def __init__(self, input_channel=3,num_clusters=800,features_len=1024,embedding_dimension=10,bias=True):
#         super(PointNet, self).__init__()
#         self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv1d(128, features_len, kernel_size=1, bias=False)
#         #self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
#         #self.conv41 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
#         #self.conv5 = nn.Conv1d(128, features_len, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(features_len)
#         #self.bn4 = nn.BatchNorm1d(128)
#         #self.bn41= nn.BatchNorm1d(128)
#         #self.bn5 = nn.BatchNorm1d(features_len)
#         self.fc1 = nn.Linear(features_len, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn11 = nn.BatchNorm1d(512)
#         self.bn12 = nn.BatchNorm1d(256)
#         self.embedding = nn.Linear(256, embedding_dimension, bias=bias)
#         self.clustering = ClusterlingLayer(embedding_dimension, num_clusters)
#         self.num_clusters = num_clusters
#         self.dropout = nn.Dropout(p=0.3)
#     def forward(self, x1,x2): #batch_size*channels*number of points
#         def forward_single(self, x):
#             #print(x.size())
#             x = F.relu(self.bn1(self.conv1(x)))
#             #print(x.size())
#             x = F.relu(self.bn2(self.conv2(x)))
#             #print(x.size())
#             x = self.bn3(self.conv3(x))
#             #print(x.size())
#             #x = F.relu(self.bn4(self.conv4(x)))
#             #x = F.relu(self.bn41(self.conv41(x)))
#             #print(x.size())
#             #x = F.relu(self.bn5(self.conv5(x)))
#             #print(x.size())
#             x = x.max(dim=-1, keepdim=False)[0]
#             x = F.relu(self.bn11(self.fc1(x)))
#             x = F.relu(self.bn12(self.dropout(self.fc2(x))))
#             #print(x.size())
#             extra_out = self.embedding(x)
#             #print(x.size())
#             clustering_out, x_dis = self.clustering(extra_out)
#             return clustering_out, extra_out, x_dis
#         clustering_out1, extra_out1,x_dis1= forward_single(self,x1)
#         clustering_out2, extra_out2,x_dis2 = forward_single(self,x2)
#         sim_score=nn.functional.pairwise_distance(extra_out1, extra_out2, p=2)
#         return sim_score, clustering_out1, clustering_out2, extra_out1,extra_out2,x_dis1
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)

class DGCNN(nn.Module):
    def __init__(self, input_channel=5,k=5):
        super(DGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp(x)
        score = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)
        return score









################################################################IVAN




















