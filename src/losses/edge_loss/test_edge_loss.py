import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import torchtestcase
import unittest
if torch.cuda.is_available():
    dtypeF = torch.cuda.FloatTensor
else:
    dtypeF = torch.FloatTensor
from edge_loss import EdgeLoss

class TestEdgeLoss(torchtestcase.TorchTestCase):
	def setUp(self):
		self.eloss = EdgeLoss()

		num_points = 10
		num_gt_points = 20
		points_dim = 3
		batch_size = 2

		self.preds = torch.rand(batch_size, num_points, points_dim)
		self.A = np.zeros((batch_size, num_points, num_points))
		self.edge_list = np.zeros((batch_size, 2, 2*num_points*num_points))
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
		self.edge_list = torch.from_numpy(self.edge_list).long()

	def verif_edge(self, pred, A):
		temp_A = Variable(torch.Tensor(A).type(dtypeF),requires_grad=False)
		
		# Might not make sense in 2D
		edges = self.regularizer(pred, temp_A)
		loss = torch.mul(edges, edges)
		loss = torch.sum(loss, dim =1)
		return torch.mean(loss[(temp_A != 0)])

	def regularizer(self, x, A):
		batch_size, num_points_x, points_dim = x.size()
		x = x.permute(0,2,1)
		x = x.repeat(1,1,num_points_x).view(batch_size, points_dim, num_points_x, num_points_x)
		x_t = x.transpose(3,2)
		x_diff = x_t - x
		
		# Filter out non-neighbours		
		A = A.unsqueeze(1)
		A = A.repeat(1,points_dim,1,1)
		x_diff = torch.mul(x_diff, A)
		return x_diff

	def test_normal(self):
		e1 = self.eloss.forward(self.preds, self.edge_list)
		e2 = self.verif_edge(self.preds, self.A)
		self.assertEqual(e1.size(), e2.size())
		self.assertAlmostEqual(e1.item(), e2.item(), 6)



if __name__ == '__main__':
	unittest.main()
