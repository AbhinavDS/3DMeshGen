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

	def test_getMetaData(self):
		max_vertices, feature_size, data_size = data_loader.getMetaData(self.params)
		self.assertEqual(max_vertices, 19)
		self.assertEqual(feature_size, 380)
		self.assertEqual(data_size, 2)
		
	
	def test_getDataLoaderBatch1(self):
		self.params.batch_size = 1	
		generator = data_loader.getDataLoader(self.params)

		train_data, train_data_normal, seq_len, proj_gt = next(generator)
		# print (train_data)
		self.assertEqual(train_data.shape[0], self.params.batch_size)
		self.assertEqual(train_data_normal.shape[0], self.params.batch_size)
		
		# train_data, train_data_normal, seq_len, proj_gt = next(generator)


	def test_getDataLoaderBatch2(self):
		self.params.batch_size = 2	
		generator = data_loader.getDataLoader(self.params)
		train_data, train_data_normal, seq_len, proj_gt = next(generator)
		self.assertEqual(train_data.shape[0], self.params.batch_size)
		self.assertEqual(train_data_normal.shape[0], self.params.batch_size)
		
if __name__ == '__main__':
	unittest.main()