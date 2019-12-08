import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../..')

from src import dtypeF, dtypeL, dtypeB

#from pdb import set_trace as brk
#https://github.com/345ishaan/DenseLidarNet/blob/master/code/chamfer_loss.py

class ChamferLoss(nn.Module):

	def __init__(self):
		super(ChamferLoss, self).__init__()

	def forward(self, preds, gts):
		P = self.batch_pairwise_dist(gts, preds.unsqueeze(0))

		# min_dist to gt from each pred point, min distance to pred from each gt point, closest point in gt from each pred point, closest point in pred from each gt point 
		dist1 = torch.min(P, 1)[0]
		dist2 = torch.min(P, 2)[0]
		idx1 = torch.min(P, 1)[1]
		idx2 = torch.min(P, 2)[1] 
		return dist1, dist2, idx1, idx2


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		diag_ind_x = torch.arange(0, num_points_x).type(dtypeL)
		diag_ind_y = torch.arange(0, num_points_y).type(dtypeL)
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P

	@staticmethod
	def getChamferLoss(dist1, dist2):
		return torch.mean(dist1) + torch.mean(dist2)