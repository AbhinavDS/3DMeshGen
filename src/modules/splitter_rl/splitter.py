import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from src import dtypeF, dtypeL, dtypeB
from torch_geometric.utils import sort_edge_index

class Splitter:
	def __init__(self):
		# Initialise Boolean flags for reward relevant to intersections with pred
		# Find boolean flags for reward relevant to intersections with gt inside reward function

		
	def replace_edge(edge_index, edge1, edge2):
		#((a[0] == e[0]) & (a[1] == e[1])).unsqueeze(0).expand(2,-1)
		assert(((edge_index[0] == edge1[0]) & (edge_index[1] == edge1[1])).sum().item() == 1)
		edge_index[((edge_index[0] == edge1[0]) & (edge_index[1] == edge1[1])).unsqueeze(0).expand(2,-1)] = edge2
		return edge_index

	def split(self, data, points):
		x, c, pid = data
		intersect_pred_l, intersect_pred_r = self.get_intersection(c.x, c.edge_index, points)
		# Number of intersections should be 2
		if intersect_pred_l is None or intersect_pred_r is None:
			return data		

		# Edge0 has both vertices on same polygon; same for edge1
		assert(pid.x[intersect_pred_l[0]].item() == pid.x[intersect_pred_r[0]].item())
		assert(pid.x[intersect_pred_l[1]].item() == pid.x[intersect_pred_r[1]].item())

		# Both edges should be in same polygon
		if pid.x[intersect_pred_l[0]].item() != pid.x[intersect_pred_l[1]].item():
			return data

		curr_pid = pid.x[intersect_pred_l[0]].item()
		# Get mask for vertices in the intersecting polygon on left -  assume line < 0
		l_mask = (self.line(p1,q1,p2,q2,c.x[:,0],c.x[:,1]) < 0) & (pid.x.squeeze() == curr_pid)
		r_mask = (self.line(p1,q1,p2,q2,c.x[:,0],c.x[:,1]) > 0) & (pid.x.squeeze() == curr_pid)

		# Both new polygons should have more than 2 vertices
		if l_mask.sum().item() <=2 or r_mask.sum().item() <=2:
			return data

		
		# Begin split
		
		# Remove old edge and replace with new
		old_edge1 = torch.Tensor([intersect_pred_l[0], intersect_pred_r[0]]).type_as(c.edge_index).unsqueeze(1)
		old_edge1_r = torch.Tensor([intersect_pred_r[0], intersect_pred_l[0]]).type_as(c.edge_index).unsqueeze(1)
		old_edge2 = torch.Tensor([intersect_pred_l[1], intersect_pred_r[1]]).type_as(c.edge_index).unsqueeze(1)
		old_edge2_r = torch.Tensor([intersect_pred_r[1], intersect_pred_l[1]]).type_as(c.edge_inde.unsqueeze(1)x).unsqueeze(1)
		new_edge1 = torch.Tensor([intersect_pred_l[0], intersect_pred_l[1]]).type_as(c.edge_index).unsqueeze(1)
		new_edge1_r = torch.Tensor([intersect_pred_l[1], intersect_pred_l[0]]).type_as(c.edge_inde.unsqueeze(1)x).unsqueeze(1)
		new_edge1 = torch.Tensor([intersect_pred_r[0], intersect_pred_r[1]]).type_as(c.edge_index).unsqueeze(1)
		new_edge2_r = torch.Tensor([intersect_pred_r[1], intersect_pred_r[0]]).type_as(c.edge_index).unsqueeze(1)
		c.edge_index = replace_edge(c.edge_index, old_edge1, new_edge1)
		c.edge_index = replace_edge(c.edge_index, old_edge1_r, new_edge1_r)
		c.edge_index = replace_edge(c.edge_index, old_edge2, new_edge2)
		c.edge_index = replace_edge(c.edge_index, old_edge2_r, new_edge2_r)
		c.edge_index = sort_edge_index(c.edge_index)
		x.edge_index = c.edge_index.clone()
		pid.edge_index = c.edge_index.clone()

		#update pid
		max_pid = pid.x.max().item()
		pid.x.squeeze()[l_mask] = max_pid + 1

		data = (x, c, pid)

		return data


	def split_and_reward(self, data, action, gt, gt_edges):
		points, done = action[:-1], action[-1]
		data = self.split(data, points)
		
		intersect_gt_l, intersect_gt_r = self.get_intersection(gt, gt_edges, points)

	def get_intersection(self, c, edge_index, points):

		edge_index_l, edge_index_r = edge_index

		x1, y1 = c[edge_index_l]
		x2, y2 = c[edge_index_r]

		[p1, q1, p2, q2] = points.tolist()

		intersect_mask = (self.line(p1,q1,p2,q2,x1,y1)*self.line(p1,q1,p2,q2,x2,y2) < 0) & (self.line(x1,y1,x2,y2,p1,q1)*self.line(x1,y1,x2,y2,p2,q2) < 0) & (self.line(p1,q1,p2,q2,x1,y1) > 0)
		if intersect_mask.sum().item() != 2:
			return None, None
		else:
			return edge_index_l[intersect_mask], edge_index_r[intersect_mask]

	def line(self, p1,q1,p2,q2,x1,y1):
			return (p2-p1)*y1 - (q2-q1)*x1 -(q1*p2-q2*p1)






		