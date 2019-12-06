import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv as GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, weights_init = 'zero'):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_dim, hidden_dim)
        self.conv2 = GConv(hidden_dim, in_dim)
        self.activation = F.relu
        self.dropout = nn.Dropout()
        #self.activation = nn.Tanh()

        if weights_init == 'zero':
            self.zero_init()

    def zero_init(self):
        nn.init.constant_(self.conv1.weight,0)
        nn.init.constant_(self.conv2.weight,0)

    def forward(self, x, edge_index):
        x = self.dropout(self.activation(self.conv1(x, edge_index)))
        x = self.dropout(self.activation(self.conv2(x, edge_index)))
        return (inputs + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, feature_size, dim_size, depth, weights_init = 'zero'):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(feature_size = feature_size, adj_mat=adj_mat, activation=activation, weights_init = weights_init)
                           for _ in range(depth)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(feature_size, feature_size)
        self.conv2 = GConv(feature_size, dim_size)
        self.activation = F.relu
        self.dropout = nn.Dropout()
        #self.activation = nn.Tanh()
      	if weights_init == 'zero':
            self.zero_init()
    def zero_init(self):
        nn.init.constant_(self.conv1.weight,0)
        nn.init.constant_(self.conv2.weight,0)

    def forward(self, inputs):
    	x, edge_index = inputs.x, inputs.edge_index
        x = self.dropout(self.activation(self.conv1(x, edge_index)))
        x = self.blocks(x, edge_index)
        c = self.conv2(x, edge_index)

        return x, c