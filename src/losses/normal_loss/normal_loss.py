import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../..')

from src import dtypeF, dtypeL, dtypeB

class NormalLoss(nn.Module):

	def __init__(self):
		super(NormalLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, pred, nearest_gt_idx, gt_normals, edge_list):
		num_points, points_dim = pred.size()
		edge_mask = (edge_list[0] != 0) | (edge_list[1] != 0) # num_edges
		nearest_gt_normal = torch.gather(gt_normals, 1, nearest_gt_idx.unsqueeze(2).expand(-1,-1,points_dim)).squeeze(0) #num_vertices x points_dim
		nearest_gt_normal = F.normalize(torch.gather(nearest_gt_normal, 0, edge_list[0].unsqueeze(1).expand(-1,points_dim)), dim = 1) #num_edges x points_dim
		edges = F.normalize(torch.gather(pred, 0, edge_list[0].unsqueeze(1).expand(-1,points_dim)) - torch.gather(pred, 0, edge_list[1].unsqueeze(1).expand(-1,points_dim)), dim=1) #num_edges x points_dim
		
		loss = torch.sum(edges*nearest_gt_normal, 1)**2
		return torch.mean(loss[edge_mask])