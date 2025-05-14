import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
from layers import GraphConvolution

class MetaEstimator(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=True):
        super(MetaEstimator, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)]).cuda()
            self.gc.append(nn.Linear(nhid,nhid).cuda())
            self.gc.append(nn.BatchNorm1d(nhid).cuda())
            # for i in range(n_in - 1):
            #     self.gc.append(GraphConvolution(nhid, nhid).cuda())
            #     self.gc.append(nn.Linear(nhid,nhid).cuda)
            #    self.gc.append(nn.BatchNorm1d(nhid).cuda)
        else:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)])
            self.gc.append(nn.Linear(nhid,nhid))
            self.gc.append(nn.BatchNorm1d(nhid))
            # for i in range(n_in - 1):
            #     self.gc.append(GraphConvolution(nhid, nhid))
            #     self.gc.append(nn.Linear(nhid,nhid))
            #     self.gc.append(nn.BatchNorm1d(nhid))
        
        # self.n_in = n_in
        # self.n_out = n_out

        if cuda:

            # self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
            # self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid).cuda() for i in range(n_out)])
            self.out_t01 = GraphConvolution(nhid,1).cuda()
            self.out_t11 = GraphConvolution(nhid,1).cuda()

        else:
            # self.out_t00 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            # self.out_t10 = nn.ModuleList([nn.Linear(nhid,nhid) for i in range(n_out)])
            self.out_t01 = GraphConvolution(nhid,1)
            self.out_t11 = GraphConvolution(nhid,1)

        self.dropout = dropout
        
        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z
        
        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        # for i in range(self.n_in):
        #     rep = F.relu(self.gc[i](rep, adj))
        #     rep = F.dropout(rep, self.dropout, training=self.training)

        rep1 = torch.cat((rep, neighborAverageT.reshape(-1, 1)), 1)
        # for i in range(self.n_out):
        #     y00 = F.relu(self.out_t00[i](rep1))
        #     y00 = F.dropout(y00, self.dropout, training=self.training)
        #     y10 = F.relu(self.out_t10[i](rep1))
        #     y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(rep1, adj).view(-1)
        y1 = self.out_t11(rep1, adj).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep1,neighborAverageT
