import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from src.loss.chamfer_loss import ChamferLoss
from src.loss.edge_loss import EdgeLoss
from src.loss.laplacian_loss import LaplacianLoss
from src.loss.normal_loss import NormalLoss
from src.modules.gbottleneck import GBottleNeck
from src.modules.gprojection import GProjection
from src.modules.vertex_adder import VertexAdder
from src.util import utils
from src import dtype, dtypeL, dtypeB

class DeformerBlock(nn.Module):
	def __init__(self, params, num_gbs, initial_adders, embed, weights_init='zero', residual_change=False):
		super(DeformerBlock, self).__init__()
		self.params = params
		self.num_gbs = num_gbs
		self.initial_adders = initial_adders
		self.embed = embed
		self.residual_change = residual_change
		assert (self.num_gbs > 0, "Number of gbs is 0")
		
		self.deformer_block = [GBottleNeck(self.params.feature_size, self.params.dim_size, self.params.depth, weights_init=weights_init).cuda() for _ in range(self.num_gbs)]
		self.adder = VertexAdder(params.add_prob).cuda()
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

	def forward(self, x, c, image_features, A, Pid, gt, gtnormals, mask):
		"""
		Args:
			c: coordinates
			x: features
			image_features:
			A:
			Pid:
			gt:
			gtnormals:
			mask:
		"""
		self.set_loss_to_zero()
		total_blocks = self.initial_adders + self.num_gbs

		for _ in range(self.initial_adders):
			x, c, A, Pid, image_features = self.adder.forward(x, c, A, Pid, image_features)
		
		if self.embed:
			x = self.deformer_block[0].embed(c)

		for gb in range(self.num_gbs):
			if gb + self.initial_adders < self.num_gbs:
				x, c, A, Pid, image_features = self.adder.forward(x, c, A, Pid, image_features)

			c_prev = c
			fetched_feature = self.projection(c, image_features)
			x, c_out = self.deformer_block[gb].forward(torch.cat((x,fetched_feature), dim = -1), A)
			if self.residual_change:
				c = c + c_out
			norm = c.size(1) * (self.num_gbs)
			self.laploss += (self.criterionL(c_prev, c, A)/norm)
			self.closs += (self.criterionC(c, gt, mask)/norm)
			self.eloss += (self.criterionE(c, A)/norm)
			self.nloss += (self.criterionN(c, gt, gtnormals, A, mask)/norm)
		self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
		proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, self.params)
		return x, c, image_features, A, Pid, proj_pred