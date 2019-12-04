import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
    dtypeF = torch.cuda.FloatTensor
else:
    dtypeF = torch.FloatTensor

class NormalLoss(nn.Module):

	def __init__(self):
		super(NormalLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, preds, nearest_gt, gt_normals, edge_list):
		batch_size, num_points, points_dim = preds.size()
		edge_mask = (edge_list[:,0] != 0) | (edge_list[:,1] != 0) # batch_size x num_edges
		nearest_gtn = torch.gather(gt_normals, 1, nearest_gt.unsqueeze(2).expand(-1,-1,points_dim)) #batch_size x num_vertices x points_dim
		nearest_gtn = F.normalize(torch.gather(nearest_gtn, 1, edge_list[:,0].unsqueeze(2).expand(-1,-1,points_dim)), dim = 2) #batch_size x num_edges x points_dim
		edges = F.normalize(torch.gather(preds, 1, edge_list[:,0].unsqueeze(2).expand(-1,-1,points_dim)) - torch.gather(preds, 1, edge_list[:,1].unsqueeze(2).expand(-1,-1,points_dim)), dim=2) #batch_size x num_edges x points_dim
		
		loss = torch.sum(edges*nearest_gtn, 2)**2
		return torch.mean(loss[edge_mask])