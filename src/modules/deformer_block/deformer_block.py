import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from src.losses.chamfer_loss.chamfer_loss import ChamferLoss
from src.losses.edge_loss.edge_loss import EdgeLoss
from src.losses.laplacian_loss.laplacian_loss import LaplacianLoss
from src.losses.normal_loss.normal_loss import NormalLoss
from .gbottleneck import GBottleNeck
from .gprojection import GProjection
from src.modules.vertex_adder.vertex_adder import VertexAdder
from src import dtypeF, dtypeL, dtypeB

class DeformerBlock(nn.Module):
	def __init__(self, params, num_gbs, initial_adders, embed, weights_init='zero', residual_change=False):
		super(DeformerBlock, self).__init__()
		self.params = params
		self.num_gbs = num_gbs
		self.initial_adders = initial_adders
		self.embed = embed
		self.embed_layer = nn.Linear(self.params.dim_size, self.params.feature_size)
		self.a = nn.ReLU() # nn.Tanh()
		self.residual_change = residual_change
		assert (self.num_gbs > 0, "Number of gbs is 0")
		
		self.deformer_block = [GBottleNeck(self.params.feature_size, self.params.dim_size, self.params.depth, weights_init=weights_init).cuda() for _ in range(self.num_gbs)]
		self.adder = VertexAdder().cuda()
		self.projection = GProjection(self.params.feature_size, self.params.dim_size, weights_init = weights_init)

		self.criterionC = ChamferLoss()
		self.criterionN = NormalLoss()
		self.criterionL = LaplacianLoss()
		self.criterionE = EdgeLoss()
		self.set_loss_to_zero()	
		
		if weights_init == 'xavier':
			nn.init.xavier_uniform_(self.embed_layer.weight)
		elif weights_init == 'zero':
			nn.init.constant_(self.embed_layer.weight,0)


		
	def __getitem__(self, key):
		return self.deformer_block[key]

	def __len__(self):
		return len(self.deformer_block)
		
	def set_loss_to_zero(self):
		self.loss = 0.0
		self.closs = 0.0
		self.laploss = 0.0
		self.nloss = 0.0
		self.eloss = 0.0

	def forward(self, batch_x, batch_c, image_features, Pid, gt, gtnormals):
		"""
		Args:
			c: coordinates
			x: features
			image_features:
			A:
			Pid:
			gt:
			gtnormals:
		"""
		self.set_loss_to_zero()
		total_blocks = self.initial_adders + self.num_gbs

		for _ in range(self.initial_adders):
			batch_x, batch_c, Pid = self.adder.forward(batch_x, batch_c, Pid)
		
		if self.embed:
			batch_x.x = self.a(self.embed_layer(batch_c.x))

		for gb in range(self.num_gbs):
			if gb + self.initial_adders < self.num_gbs:
				batch_x, batch_c, Pid = self.adder.forward(batch_x, batch_c, Pid)

			c_prev = batch_c.x
			fetched_feature = self.projection(batch_c.x, image_features)
			batch_x.x = torch.cat((batch_x.x,fetched_feature), dim = -1)
			batch_x.x, c_out = self.deformer_block[gb].forward(batch_x)
			if self.residual_change:
				batch_c.x = batch_c.x + c_out
			else:
				batch_c.x = c_out
			
			c = batch_c.x
			self.laploss += self.criterionL(c_prev, c, batch_c.edge_index)
			dist1, dist2, idx1, _ = self.criterionC(c, gt)
			self.closs += ChamferLoss.getChamferLoss(dist1, dist2)
			self.eloss += self.criterionE(c, batch_c.edge_index)
			self.nloss += self.criterionN(c, idx1, gtnormals, batch_c.edge_index)
		self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
		return batch_x, batch_c, Pid