"""
Module that builds the Pixel2Mesh
"""

import torch 
from torch import nn as nn
import numpy as np
from torch.nn.utils.weight_norm import weight_norm as wn

from torch_geometric.data import Data, Batch
from src.modules.deformer_block.deformer_block import DeformerBlock
from src import dtypeF, dtypeL, dtypeB

class Pixel2Mesh(nn.Module):

	def __init__(self, params):

		super(Pixel2Mesh, self).__init__()
		self.device = params.device
		self.params = params

		self.db1 = DeformerBlock(self.params, self.params.gbottlenecks, self.params.initial_adders, False, weights_init='xavier', residual_change=False)
		# db2 = DeltaDeformerBlock(self.params, self.params.num_gcns2, 0, False, residual_change=True)

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

	def forward(self, image_features, gt, gtnormals):
		init_batch_x, init_batch_c, init_batch_pid = self.create_start_data()
		batch_x1, batch_c1, batch_pid = self.db1.forward(init_batch_x, init_batch_c, image_features, init_batch_pid, gt, gtnormals)
		self.closs = self.db1.closs
		self.nloss = self.db1.nloss
		self.eloss = self.db1.eloss
		self.laploss = self.db1.laploss
		self.loss = self.db1.loss

		return batch_x1, batch_c1
		