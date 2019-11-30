"""
Tests for laplacian_loss.py file.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torchtestcase
import unittest

import laplacian_loss

if torch.cuda.is_available():
    dtypeF = torch.cuda.FloatTensor
    dtypeL = torch.cuda.LongTensor
else:
    dtypeF = torch.FloatTensor
    dtypeL = torch.LongTensor


class TestLaplacianLoss(torchtestcase.TorchTestCase):
	def setUp(self):
		"""
		Initial setup for all the tests.
		"""

		self.loss = laplacian_loss.LaplacianLoss()
		random.seed(1)
		np.random.seed(0)
		torch.manual_seed(0)

		self.batch_size = 5
		self.dim_size = 3
		self.num_vertices = 10
		self.max_edges = 10
		self.num_edges = 30
		
		# Generate Data
		self.A_list = torch.zeros((self.batch_size, self.num_vertices, self.max_edges)).type(dtypeL)-1
		for batch in range(self.batch_size):
			edges = set()
			edge_counter = 0
			counter = [0 for i in range(self.num_vertices)]
			while edge_counter < self.num_edges:
				u,v = random.sample(range(self.num_vertices), 2)
				if (u,v) not in edges and counter[u] < self.max_edges and counter[v] < self.max_edges:
					edges.add((u,v))
					edges.add((v,u))
					counter[u] += 1
					counter[v] += 1
					edge_counter += 1
				else:
					continue

			for u,v in edges:
				counter[u] -= 1
				self.A_list[batch][u][counter[u]] = v
		
	def test_centroid(self):
		coord = torch.rand((self.batch_size, self.num_vertices, self.dim_size))
		
		# Generate Expected Results
		correct = torch.zeros_like(coord)
		for batch in range(self.batch_size):
			for u in range(self.num_vertices):
				num_neighbours = 0
				for v in self.A_list[batch][u]:
					if v == -1:
						break
					num_neighbours += 1
					correct[batch][u] += coord[batch][v]
				correct[batch][u] /= num_neighbours


		# Check centroid generation
		results = self.loss.centroid(coord, self.A_list)

		# If exact answers are generated, can use Equal for tensors. (Not Almost Equal)
		self.assertEqual(correct.size(), results.size())
		self.assertEqual(correct, results)
		



	def test_forward(self):
		coord1 = torch.rand((self.batch_size, self.num_vertices, self.dim_size))
		coord2 = torch.rand((self.batch_size, self.num_vertices, self.dim_size))
		
		# Generate Expected Results
		lap_coord1 = coord1 - self.loss.centroid(coord1, self.A_list)
		lap_coord2 = coord2 - self.loss.centroid(coord2, self.A_list)
		correct = torch.sum((lap_coord2 - lap_coord1)**2)/(self.batch_size * self.dim_size)

		# Check loss generation
		results = self.loss.forward(coord1, coord2, self.A_list.cpu().numpy())
		
		# For single value results, convert to item and use AlmostEqual
		self.assertAlmostEqual(correct.item(), results.item(), 6)

if __name__ == '__main__':
	unittest.main()