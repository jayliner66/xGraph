import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # for parameter in self.gc1.parameters():
        #     print(parameter.size())
        self.gc2 = GraphConvolution(nhid, nclass)
        # for parameter in self.gc2.parameters():
        #     print(parameter.size())
        self.dropout = dropout

    def forward(self, x, adj, x_row = None, adj_row = None, i = None):
        if x_row is not None:
            x[i][:] = x_row
            adj[i][:] = adj_row
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
