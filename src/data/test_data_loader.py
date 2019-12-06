"""
Tests for data_loader.py file.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torchtestcase
import unittest
import argparse

import data_loader

if torch.cuda.is_available():
    dtypeF = torch.cuda.FloatTensor
    dtypeL = torch.cuda.LongTensor
else:
    dtypeF = torch.FloatTensor
    dtypeL = torch.LongTensor


class TestDataLoader(torchtestcase.TorchTestCase):
	def setUp(self):
		"""
		Setup for all the tests.
		"""
		params = argparse.ArgumentParser().parse_args()
		params.data_dir = 'unittest_data'
		params.suffix = 'unittest'
		params.dim_size = 2
		params.feature_scale = 10
		params.img_width = 600
		self.params = params
		data_loader.seedRandom(15)

	def test_getMetaData(self):
		max_vertices, feature_size, data_size, max_total_vertices = data_loader.getMetaData(self.params)
		self.assertEqual(max_vertices, 19)
		self.assertEqual(feature_size, 380)
		self.assertEqual(data_size, 2)
		self.assertEqual(max_total_vertices, 38)
	
	def test_upsample(self):
		num_vertices = 11
		dim_size = 2
		vertices = np.random.rand(num_vertices, dim_size)
		normals = np.random.rand(num_vertices, dim_size)
		normals /= np.expand_dims(np.linalg.norm(normals, axis=1), 1)
		unidirection_edges = [(0,1), (1,2), (2,3), (3,9), (9,0), (4,5), (5,6), (6,4), (7,8), (8,10), (10,7)]
		edges = set()
		for a,b in unidirection_edges:
			edges.add((a,b))
			edges.add((b,a))
		
		max_vertices = 15
		new_vertices, new_normals, new_edges = data_loader.upsample(vertices, normals, edges, max_vertices)

		# Order matters for checking as vertex id assigned accordingly
		predecided_edges_removed = [(3, 9), (8, 10), (2, 1), (1, 0)] 
		check_vertices = np.zeros((max_vertices, dim_size))
		check_normals =  np.zeros((max_vertices, dim_size))
		check_vertices[0:num_vertices , :] = vertices
		check_normals[0:num_vertices , :] = normals
		check_edges = edges.copy()
		for i, edge in enumerate(predecided_edges_removed):
				a, b = edge
				c = num_vertices + i
				check_edges.remove((a,b))
				check_edges.remove((b,a))
				check_edges.add((a,c))
				check_edges.add((c,a))
				check_edges.add((b,c))
				check_edges.add((c,b))
				check_vertices[c] = (check_vertices[a] + check_vertices[b])/2
				check_normals[c] = (check_normals[a] + check_normals[b])/2
				check_normals[c] /= np.linalg.norm(check_normals[c])
		self.assertEqual(new_edges, check_edges)
		np.testing.assert_array_equal(new_vertices, check_vertices)
		np.testing.assert_array_equal(new_normals, check_normals)
		self.assertFalse(((np.isnan(new_vertices) | np.isnan(new_normals))).any())

	def test_getPointsAndEdges(self):
		#TODO: Single Function Test
		pass

	def test_getDataLoaderBatch1(self):
		#TODO: Exact value matches for vertices, normals, edges, proj_gt

		self.params.batch_size = 1	
		max_vertices, feature_size, data_size, max_total_vertices = data_loader.getMetaData(self.params)
		generator = data_loader.getDataLoader(self.params)
		train_data, train_data_normal, edges, proj_gt = next(generator)
		self.assertEqual(train_data.shape, (self.params.batch_size, max_total_vertices, self.params.dim_size))
		self.assertEqual(train_data_normal.shape, (self.params.batch_size, max_total_vertices, self.params.dim_size))
		self.assertEqual(edges.shape, (self.params.batch_size,))
		self.assertEqual(proj_gt.shape, (self.params.batch_size, self.params.img_width))

	def test_getDataLoaderBatch2(self):
		#TODO: Exact value matches for vertices, normals, edges, proj_gt

		self.params.batch_size = 2	
		max_vertices, feature_size, data_size, max_total_vertices = data_loader.getMetaData(self.params)
		generator = data_loader.getDataLoader(self.params)
		train_data, train_data_normal, edges, proj_gt = next(generator)
		self.assertEqual(train_data.shape, (self.params.batch_size, max_total_vertices, self.params.dim_size))
		self.assertEqual(train_data_normal.shape, (self.params.batch_size, max_total_vertices, self.params.dim_size))
		self.assertEqual(edges.shape, (self.params.batch_size,))
		self.assertEqual(proj_gt.shape, (self.params.batch_size, self.params.img_width))

if __name__ == '__main__':
	unittest.main()