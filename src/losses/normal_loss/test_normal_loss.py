import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import torchtestcase
import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../..')

from src import dtypeF, dtypeL, dtypeB
from normal_loss import NormalLoss

class TestNormalLoss(torchtestcase.TorchTestCase):
	def setUp(self):
		self.nloss = NormalLoss()

		random.seed(1)
		np.random.seed(0)
		torch.manual_seed(0)
		num_points = 100
		num_gt_points = 200
		points_dim = 3
		batch_size = 16

		self.preds = torch.rand(batch_size, num_points, points_dim).type(dtypeF)
		self.gt_normals = torch.rand(batch_size, num_gt_points, points_dim).type(dtypeF)
		self.A = np.zeros((batch_size, num_points, num_points))
		self.edge_list = np.zeros((batch_size, 2, 2*num_points*num_points))
		self.nearest_gt_idx = []
		for b in range(batch_size):
			counter = 0
			for i in range(num_points):
				for j in range(i+1, num_points):
						if random.randint(0,1):
							self.A[b,i,j] = 1
							self.A[b,j,i] = 1
							self.edge_list[b,0,counter] = i
							self.edge_list[b,1,counter] = j
							counter += 1
							self.edge_list[b,0,counter] = j
							self.edge_list[b,1,counter] = i
							counter += 1

			self.nearest_gt_idx.append([random.randint(0, num_gt_points - 1) for p in range(0, num_points)])
		self.edge_list = torch.from_numpy(self.edge_list).type(dtypeL)
		self.nearest_gt_idx = torch.from_numpy(np.array(self.nearest_gt_idx)).type(dtypeL)

	def normal_pred(self, x, A):
		batch_size, num_points_x, points_dim = x.size()
		x = x.permute(0,2,1)
		x = x.unsqueeze(-1).expand(-1,-1,-1,num_points_x)
		x_t = x.transpose(3,2)
		x_diff = x_t - x
		x_diff = F.normalize(x_diff, dim = 1)

		# Filter out non-neighbours		
		A = A.unsqueeze(1).expand(-1, points_dim, -1, -1)
		x_diff = torch.mul(x_diff, A)
		return x_diff

	def calculate_loss(self, p_k, nq, mask):
		batch_size, points_dim, num_points_x, _ = p_k.size()
		nq = nq.transpose(2,1)
		nq = nq.unsqueeze(2).expand(-1,-1,-1,num_points_x)
		inner_product = torch.mul(nq, p_k)
		
		inner_product = torch.sum(inner_product, dim=1)
		inner_product = torch.mul(inner_product,inner_product)
		return torch.mean(inner_product[mask])

	def verif_forward(self, preds, nearest_gt_idx, gt_normals, A):
		temp_A = Variable(torch.Tensor(A).type(dtypeF), requires_grad=False)
		batch_size, num_points, points_dim = preds.size()
		q = F.normalize(torch.gather(gt_normals, 1, nearest_gt_idx.unsqueeze(2).expand(-1,-1,points_dim)), dim = 2)

		# Calculate difference for each pred vertex, use adj mat to filter out non-neighbours
		diff_neighbours = self.normal_pred(preds, temp_A)
		# Calculate final loss
		return self.calculate_loss(diff_neighbours, q, (temp_A != 0))
	def test_normal(self):
		e1 = self.nloss.forward(self.preds, self.nearest_gt_idx, self.gt_normals, self.edge_list)
		e2 = self.verif_forward(self.preds, self.nearest_gt_idx, self.gt_normals, self.A)
		self.assertEqual(e1.size(), e2.size())
		self.assertAlmostEqual(e1.item(), e2.item(), 6)

if __name__ == '__main__':
	unittest.main()


