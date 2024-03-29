import torch.nn as nn

# from torch_geometric.nn import GCNConv as GConv
from torch_geometric.nn import GraphConv as GConv

class GResBlock(nn.Module):

	def __init__(self, in_dim, hidden_dim, weights_init = 'normal'):
		super(GResBlock, self).__init__()

		self.conv1 = GConv(in_dim, hidden_dim)
		self.conv2 = GConv(hidden_dim, in_dim)
		#self.activation = nn.ReLU()
		self.activation = nn.Tanh()
		# self.dropout = nn.Dropout()
		self.dropout = nn.Dropout(p=0)


		if weights_init == 'zero':
			self.zero_init()

	def zero_init(self):
		nn.init.constant_(self.conv1.weight,0)
		nn.init.constant_(self.conv2.weight,0)

	def forward(self, batch_x):
		
		# x = self.dropout(self.activation(self.conv1(batch_x.x, batch_x.edge_index)))
		# x = self.dropout(self.activation(self.conv2(x, batch_x.edge_index)))
		# batch_x.x = (x + batch_x.x) * 0.5

		x = self.dropout(self.activation(self.conv1(batch_x.x, batch_x.edge_index) + batch_x.x))
		# x = self.dropout(self.activation(self.conv2(x, batch_x.edge_index) + batch_x.x))
		batch_x.x = x

		return batch_x


class GBottleNeck(nn.Module):
	def __init__(self, feature_size, dim_size, depth, weights_init = 'normal'):
		super(GBottleNeck, self).__init__()

		resblock_layers = [GResBlock(feature_size, feature_size, weights_init = weights_init) for _ in range(depth)]
		self.blocks = nn.Sequential(*resblock_layers)
		self.conv1 = GConv(2*feature_size, feature_size)
		self.conv2 = GConv(feature_size, dim_size)
		self.conv3 = nn.Linear(feature_size, dim_size)
		#self.activation = nn.ReLU()
		self.activation = nn.Tanh()
		# self.dropout = nn.Dropout()
		self.dropout = nn.Dropout(p=0)
		if weights_init == 'zero':
			self.zero_init()

	def zero_init(self):
		nn.init.constant_(self.conv1.weight,0)
		nn.init.constant_(self.conv2.weight,0)
		nn.init.constant_(self.conv3.weight,0)

	def forward(self, batch_x):
		batch_x.x = self.dropout(self.activation(self.conv1(batch_x.x, batch_x.edge_index)))
		batch_x = self.blocks(batch_x)
		c = self.conv2(batch_x.x, batch_x.edge_index)
		#c = self.activation(self.conv3(batch_x.x))
		return batch_x.x, c