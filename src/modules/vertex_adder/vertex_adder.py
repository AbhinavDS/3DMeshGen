import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtypeF, dtypeL, dtypeB
from torch_geometric.utils import to_undirected

class VertexAdder(nn.Module):
	def __init__(self):
		super(VertexAdder, self).__init__()
		pass

	def forward(self, batch_x, batch_c, batch_pid):
		"""
		x_prev: v x f
		c_prev: v x d
		pid_prev:  v x 1
		edge_index: 2 x e
		"""
		edge_index = batch_x.edge_index
		x_prev = batch_x.x.unsqueeze(0)
		c_prev = batch_c.x.unsqueeze(0)
		pid_prev = batch_pid.x.unsqueeze(0)

		_,num_edges = edge_index.size()
		_,num_nodes,feature_size = x_prev.size()
		
		
		d_mask = (edge_index[0] < edge_index[1])
		d_edge_index = torch.masked_select(edge_index, d_mask).reshape(2, -1).type_as(edge_index)
		d_edge_index_l, d_edge_index_r = d_edge_index
		
		num_d_edges = d_edge_index.size(1)
		new_num_nodes = num_d_edges + num_nodes
		new_edge_index =  torch.Tensor(edge_index.size(0), 2*num_d_edges).type_as(edge_index)

		new_edge_index[0,:num_d_edges] = d_edge_index_l
		new_edge_index[1,:num_d_edges] = torch.arange(num_nodes, new_num_nodes).type_as(edge_index)
		new_edge_index[0,num_d_edges:] = torch.arange(num_nodes, new_num_nodes).type_as(edge_index)
		new_edge_index[1,num_d_edges:] = d_edge_index_r
		new_edge_index = to_undirected(new_edge_index, new_num_nodes)

		new_x = torch.zeros(x_prev.size(0), new_num_nodes, x_prev.size(2)).type_as(x_prev)
		new_c = torch.zeros(c_prev.size(0), new_num_nodes, c_prev.size(2)).type_as(c_prev)
		new_pid = torch.zeros(pid_prev.size(0), new_num_nodes, pid_prev.size(2)).type_as(pid_prev)

		new_x[:,:num_nodes] = x_prev
		new_c[:,:num_nodes] = c_prev
		new_pid[:,:num_nodes] = pid_prev

		new_x[:,num_nodes:] = 0.5*(x_prev[:,d_edge_index_l] + x_prev[:,d_edge_index_r])
		new_c[:,num_nodes:] = 0.5*(c_prev[:,d_edge_index_l] + c_prev[:,d_edge_index_r])
		new_pid[:,num_nodes:] = pid_prev[:,d_edge_index_l]


		batch_x.x = new_x.squeeze(0)
		batch_c.x = new_c.squeeze(0)
		batch_pid.x = new_pid.squeeze(0)
		batch_x.edge_index = new_edge_index
		batch_c.edge_index = new_edge_index

		return batch_x, batch_c, batch_pid