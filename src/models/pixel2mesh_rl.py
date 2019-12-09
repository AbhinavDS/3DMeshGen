"""
Module that builds the Pixel2MeshRL
"""

import torch 
from torch import nn as nn
import numpy as np
from torch.nn.utils.weight_norm import weight_norm as wn

from torch_geometric.data import Data, Batch
from src.modules.deformer_block.deformer_block import DeformerBlock
from src.modules.deformer_block.delta_deformer_block import DeltaDeformerBlock
from src.modules.splitter_rl.rl_agent import RLAgent
from src import dtypeF, dtypeL, dtypeB

class Pixel2MeshRL(nn.Module):

	def __init__(self, params):

		super(Pixel2MeshRL, self).__init__()
		self.device = params.device
		self.params = params

		self.db1 = DeformerBlock(self.params, self.params.gbottlenecks, self.params.initial_adders, False, weights_init='xavier', residual_change=False)
		self.rl_agent = RLAgent(self.params, self.max_polygons, agent='sac')
		self.db2 = DeltaDeformerBlock(self.params, self.params.gbottlenecks, residual_change=True, weights_init = 'zero')

	def create_start_data(self):
		"""
			creates, features, coordinates and Edge list
		"""
		def inputMesh(feature_size):
			c1= np.expand_dims(np.array([0,-0.9]),0)
			c2= np.expand_dims(np.array([-0.9,0.9]),0)
			c3= np.expand_dims(np.array([0.9,0.9]),0)
			x1 = np.expand_dims(np.pad(np.array([0,-0.9]),(0,feature_size-2),'constant',constant_values=(0,0)),0)
			x2 = np.expand_dims(np.pad(np.array([-0.9,0.9]),(0,feature_size-2),'constant',constant_values=(0,0)),0)
			x3 = np.expand_dims(np.pad(np.array([0.9,0.9]),(0,feature_size-2),'constant',constant_values=(0,0)),0)
			edge_index = np.transpose(np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])) # COO format
			return np.concatenate((c1,c2,c3),axis=0), np.concatenate((x1,x2,x3),axis=0),edge_index

		c, x, edge_index = inputMesh(self.params.feature_size)# x is c with zeros appended, x=f ..pixel2mesh
		data_list_x = []
		data_list_c = []
		data_list_pid = []
		for i in range(self.params.batch_size):
			data_list_x.append(Data(x=torch.Tensor(x).type(dtypeF), edge_index=torch.Tensor(edge_index).type(dtypeL)))
			data_list_c.append(Data(x=torch.Tensor(c).type(dtypeF), edge_index=torch.Tensor(edge_index).type(dtypeL)))
			data_list_pid.append(Data(x=torch.zeros(c.shape[0],1).type(dtypeL).requires_grad_(False)))
		batch_x = Batch.from_data_list(data_list_x)
		batch_c = Batch.from_data_list(data_list_c)
		batch_pid = Batch.from_data_list(data_list_pid)
		return batch_x, batch_c, batch_pid

	def forward(self, image_features, gt, gt_normals, proj_gt, gt_edges = None):
		init_batch_x, init_batch_c, init_batch_pid = self.create_start_data()
		batch_x, batch_c, batch_pid = self.db1.forward(init_batch_x, init_batch_c, image_features, init_batch_pid, gt, gt_normals)
		if self.training:
			data = (batch_x, batch_c, batch_pid)
			batch_c = self.rl_agent.train(self.db2, data, image_features, gt, gt_normals, proj_gt, gt_edges)
		else:
			batch_c = self.rl_agent.eval(self.db2, data, image_features, proj_gt)

		self.closs = self.db1.closs + self.db2.closs
		self.nloss = self.db1.nloss + self.db2.nloss
		self.eloss = self.db1.eloss + self.db2.eloss
		self.laploss = self.db1.laploss + self.db2.laploss
		self.loss = self.db1.loss + self.db2.loss

		return batch_c
		