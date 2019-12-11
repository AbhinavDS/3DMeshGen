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

class DeltaDeformerBlock(nn.Module):
	def __init__(self, params, num_gbs, weights_init='xavier', residual_change=False):
		super(DeltaDeformerBlock, self).__init__()
		self.params = params
		self.num_gbs = num_gbs
		#self.activation = nn.ReLU()
		self.activation = nn.Tanh()
		self.residual_change = residual_change
		assert (self.num_gbs > 0, "Number of gbs is 0")
		
		self.deformer_block = nn.ModuleList([GBottleNeck(self.params.feature_size, self.params.dim_size, self.params.gcn_depth, weights_init=weights_init).cuda() for _ in range(self.num_gbs)])
		self.adder = VertexAdder().cuda()
		self.projection = GProjection(self.params.feature_size, self.params.dim_size, weights_init = weights_init)

		self.criterionC = ChamferLoss()
		self.criterionN = NormalLoss()
		self.criterionL = LaplacianLoss()
		self.criterionE = EdgeLoss()
		self.set_loss_to_zero()


		
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

	def scaleLosses(self, factor):
		self.loss *= factor
		self.closs *= factor
		self.laploss *= factor
		self.nloss *= factor
		self.eloss *= factor

	def forward(self, batch_x, batch_c, image_features, Pid, gt, gt_normals, add_loss = True):
		"""
		Args:
			c: coordinates
			x: features
			image_features:
			A:
			Pid:
			gt:
			gt_normals:
		"""
		# if add_loss:
		# 	batch_x, batch_c, Pid = self.adder.forward(batch_x, batch_c, Pid)
		for gb in range(self.num_gbs):
			c_prev = batch_c.x
			fetched_feature = self.projection(batch_c.x, image_features)
			batch_x.x = torch.cat((batch_x.x,fetched_feature), dim = -1)
			batch_x.x, c_out = self.deformer_block[gb].forward(batch_x, batch_c = batch_c)
			if self.residual_change:
				#batch_c.x = self.activation(batch_c.x + c_out)
				batch_c.x = batch_c.x + c_out
			else:
				batch_c.x = c_out
			
			c = batch_c.x
			factor = 1#self.num_gbs - gb
			if add_loss:
				self.laploss += self.criterionL(c_prev, c, batch_c.edge_index) * factor
				dist1, dist2, idx1, _ = self.criterionC(c, gt)
				self.closs += ChamferLoss.getChamferLoss(dist1, dist2) * factor
				self.eloss += self.criterionE(c, batch_c.edge_index) * factor
				self.nloss += self.criterionN(c, idx1, gt_normals, batch_c.edge_index) * factor
				self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
		return batch_x, batch_c, Pid