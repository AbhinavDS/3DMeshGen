"""
Module to calculate laplacian regularization of the vertices after deformation, to prevent too rapid deformation.
"""

import torch
import torch.nn as nn
import os
import sys
from torch_geometric.utils import sort_edge_index
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../..')

from src import dtypeF, dtypeL, dtypeB

class LaplacianLoss(nn.Module):

	def __init__(self, use_edge_index=True):
		super(LaplacianLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()
		self.l2_loss = nn.MSELoss(reduction='mean')   
		self.use_edge_index = use_edge_index
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
	def getAdjacencyList(self, A_list):
		if self.use_edge_index:
			temp_A_list = A_list.transpose(0,1).reshape(-1, 4)[:,[1,3]].unsqueeze(0).requires_grad_(False)
		else:
			temp_A_list = torch.Tensor(A_list).type(dtypeL).requires_grad_(False)
		return temp_A_list




	def forward(self, c1, c2, edge_index):
		"""
		Args:
			c1: Original position of different points (batch_size x num_points x points_dim)
			c2: Final Position of different points (batch_size x num_points x points_dim)
			edge_index: [2xe] which we convert to adjanceny List or A_list

		Returns:
			loss: Laplacian regularlised loss.
		"""
		if len(c1.size()) < 3:
			coord1 = c1.unsqueeze(0)
			coord2 = c2.unsqueeze(0)
		else:
			coord1 = c1
			coord2 = c2

		temp_A_list = self.getAdjacencyList(edge_index)
		lap_coord1 = coord1 - self.centroid(coord1, temp_A_list)
		lap_coord2 = coord2 - self.centroid(coord2, temp_A_list)
		loss = self.l2_loss(lap_coord2, lap_coord1)# * coord1.size(1)
		return loss

	def centroid(self, coord, A_list):
		"""
		Args:
			coord: Position of different points (batch_size x num_points x points_dim)
			A_list: Adjacency List (batch_size x num_points x 10): contains invalid indices 

		Returns:
			centroid: Centroid calculated by checking neighbouring poistions.
		"""
		dim_size, edge_size = coord.size(2), A_list.size(2)

		valid_mask =  A_list >= 0
		valid_indices = A_list.clone()
		
		valid_indices[~valid_mask] = 0
		x = coord.unsqueeze(3).expand(*coord.size(), edge_size).permute(0,1,3,2)
		valid_indices = valid_indices.unsqueeze(3).expand((*A_list.size(), dim_size))
		vertices = torch.gather(x, 1, valid_indices)
		valid_mask = valid_mask.unsqueeze(3).expand(*valid_mask.size(), dim_size)
		sum_neighbours = torch.sum(vertices*valid_mask, 2)
		num_neighbours = torch.sum((A_list>=0).type(dtypeF), dim=2).unsqueeze(-1)
		centroid = torch.div(sum_neighbours, num_neighbours)

		return centroid
