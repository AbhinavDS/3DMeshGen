"""
Module that builds the Pixel2Mesh
"""

import torch 
from torch import nn as nn
from torch.nn.utils.weight_norm import weight_norm as wn

from torch_geometric.data import Data, Batch
from ..modules.deformer_block import DeformerBlock
from  src import dtypeF, dtypeL, dtypeB

class Pixel2Mesh(nn.Module):

	def __init__(self, params):

		super(Pixel2Mesh, self).__init__()
		self.device = args.device
		self.params = params

		db1 = DeformerBlock(self.params, self.params.num_gcns1, self.params.initial_adders, True, weights_init='xavier')
		# db2 = DeltaDeformerBlock(self.params, self.params.num_gcns2, 0, False, residual_change=True)

	def create_start_data():
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
			A = np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]) # COO format
			return np.concatenate((c1,c2,c3),axis=0), np.concatenate((x1,x2,x3),axis=0),A

		c, x, A = utils.inputMesh(self.params.feature_size)# x is c with zeros appended, x=f ..pixel2mesh
		data_list = []
		for _ in range(self.params.batch_size):
			data_list.append(Data(x=torch.Tensor(x).type(dtypeF), pos=torch.Tensor(c).type(dtypeF), edge_index=torch.Tensor(A).type(dtypeL)))
		batch = Batch.from_data_list(data_list)
		return batch

	def forward(self, image_features, gt, gtnormals, mask):
		batch = self.create_start_data()
		# change deformer to take batch as input and output batch as well
		x, c, s, A, proj_pred = db1.forward(x, c, image_features, A, Pid, gt, gtnormals, mask)
		self.closs = db1.closs
		self.nloss = db1.nloss
		self.eloss = db1.eloss
		self.laploss = db1.laploss
		self.loss = db1.loss
		