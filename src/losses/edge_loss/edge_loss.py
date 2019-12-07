import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../..')

from src import dtypeF, dtypeL, dtypeB
    
class EdgeLoss(nn.Module):

	def __init__(self):
		super(EdgeLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, pred, edge_list):
		num_points, points_dim = pred.size()

		edge_mask = (edge_list[0] != 0) | (edge_list[1] != 0) # num_edges
		edges = torch.gather(pred, 0, edge_list[0].unsqueeze(1).expand(-1,points_dim)) - torch.gather(pred, 0, edge_list[1].unsqueeze(1).expand(-1,points_dim)) #num_edges x points_dim
		loss = torch.sum(edges**2, dim = 1)
		return torch.mean(loss[edge_mask])