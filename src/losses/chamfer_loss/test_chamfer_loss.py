import torch
import torch.nn as nn
import torch.nn.functional as F
from chamfer_loss import ChamferLoss
import random
import numpy as np
import torchtestcase
import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../..')

from src import dtypeF, dtypeL, dtypeB



class TestChamferLoss(torchtestcase.TorchTestCase):
	def setUp(self):
		self.chamfer = ChamferLoss()
		random.seed(1)
		np.random.seed(0)
		torch.manual_seed(0)

	def test_chamfer(self):
		pass

if __name__ == '__main__':
	unittest.main()



