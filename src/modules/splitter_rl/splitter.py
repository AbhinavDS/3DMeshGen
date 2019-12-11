import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from src import dtypeF, dtypeL, dtypeB
from torch_geometric.utils import sort_edge_index

class Splitter:
	def __init__(self, params):
		self.flags = {}
		self.params = params
		# Initialise Boolean flags for reward relevant to intersections with pred
		# Find boolean flags for reward relevant to intersections with gt inside reward function

		
	def replace_edge(self, edge_index, edge1, edge2):
		#((a[0] == e[0]) & (a[1] == e[1])).unsqueeze(0).expand(2,-1)
		assert(((edge_index[0] == edge1[0]) & (edge_index[1] == edge1[1])).sum().item() == 1)

		edge_index[((edge_index[0] == edge1[0]) & (edge_index[1] == edge1[1])).unsqueeze(0).expand(2,-1)] = edge2.squeeze(1)
		return edge_index

	def split(self, data, action):
		x, c, pid = data
		points, done = action[:-1], action[-1]
		done = (done > 0)
		intersect_pred_l, intersect_pred_r, num_intersect = self.get_intersection(c.x, c.edge_index, points)
		self.flags['num_intersections'] = num_intersect
		# Number of intersections should be 2
		if intersect_pred_l is None or intersect_pred_r is None:
			return data, done

		# Edge0 has both vertices on same polygon; same for edge1
		assert(pid.x[intersect_pred_l[0]].item() == pid.x[intersect_pred_r[0]].item())
		assert(pid.x[intersect_pred_l[1]].item() == pid.x[intersect_pred_r[1]].item())

		# Both edges should be in same polygon
		if pid.x[intersect_pred_l[0]].item() != pid.x[intersect_pred_l[1]].item():
			self.flags['diff_polygons'] = True
			return data, done

		curr_pid = pid.x[intersect_pred_l[0]].item()
		# Get mask for vertices in the intersecting polygon on left -  assume line < 0
		p1,q1,p2,q2 = points
		l_mask = (self.line(p1,q1,p2,q2,c.x[:,0],c.x[:,1]) < 0) & (pid.x.squeeze() == curr_pid)
		r_mask = (self.line(p1,q1,p2,q2,c.x[:,0],c.x[:,1]) > 0) & (pid.x.squeeze() == curr_pid)

		# Both new polygons should have more than 2 vertices
		if l_mask.sum().item() <=2 or r_mask.sum().item() <=2:
			self.flags['degenerate_split'] = True
			return data, done


		
		# Begin split
		
		# Remove old edge and replace with new
		old_edge1 = torch.Tensor([intersect_pred_l[0], intersect_pred_r[0]]).type_as(c.edge_index).unsqueeze(1)
		old_edge1_r = torch.Tensor([intersect_pred_r[0], intersect_pred_l[0]]).type_as(c.edge_index).unsqueeze(1)
		old_edge2 = torch.Tensor([intersect_pred_l[1], intersect_pred_r[1]]).type_as(c.edge_index).unsqueeze(1)
		old_edge2_r = torch.Tensor([intersect_pred_r[1], intersect_pred_l[1]]).type_as(c.edge_index).unsqueeze(1)
		new_edge1 = torch.Tensor([intersect_pred_l[0], intersect_pred_l[1]]).type_as(c.edge_index).unsqueeze(1)
		new_edge1_r = torch.Tensor([intersect_pred_l[1], intersect_pred_l[0]]).type_as(c.edge_index).unsqueeze(1)
		new_edge2 = torch.Tensor([intersect_pred_r[0], intersect_pred_r[1]]).type_as(c.edge_index).unsqueeze(1)
		new_edge2_r = torch.Tensor([intersect_pred_r[1], intersect_pred_r[0]]).type_as(c.edge_index).unsqueeze(1)

		c.edge_index = self.replace_edge(c.edge_index, old_edge1, new_edge1)
		c.edge_index = self.replace_edge(c.edge_index, old_edge1_r, new_edge1_r)
		c.edge_index = self.replace_edge(c.edge_index, old_edge2, new_edge2)
		c.edge_index = self.replace_edge(c.edge_index, old_edge2_r, new_edge2_r)

		c.edge_index = sort_edge_index(c.edge_index)[0].clone().detach()
		x.edge_index = c.edge_index.clone().detach()
		pid.edge_index = c.edge_index.clone().detach()

		c.x = c.x.clone().detach()
		x.x = x.x.clone().detach()
		pid.x = pid.x.clone().detach()
		
		#update pid
		max_pid = pid.x.max().item()
		pid.x.squeeze()[l_mask] = max_pid + 1

		data = (x, c, pid)

		return data, done


	def split_and_reward(self, data, action, gt, gt_edges, gt_num_polygons):
		reward = 0.0
		points, done = action[:-1], action[-1]
		done = (done > 0)
		max_pred_pid = data[2].x.max().item()
		if not done:
			# Agent says not done and it is actually not done
			if max_pred_pid + 1 < gt_num_polygons.squeeze(axis = 0):
				reward += 0#20
			# Agent says not done but it is actually done
			else:
				reward += -20
		else:
			# Agent says done but it is actually not done
			if max_pred_pid + 1 < gt_num_polygons.squeeze(axis = 0):
				reward += -40
			# Agent says done and it is actually done
			else:
				reward += 0#20

		#print(done, reward)
		self.flags = {'diff_polygons': False, 'num_intersections': 0, 'degenerate_split': False}
		data, done = self.split(data, action)
		_, _, num_intersect_gt = self.get_intersection(gt.squeeze(0), gt_edges.squeeze(0).transpose(0,1), points)
		gt_intersect = (num_intersect_gt > 0)


		pred_bounds = data[1].x[:,0].min().item(), data[1].x[:,1].min().item(), data[1].x[:,0].max().item(), data[1].x[:,1].max().item()
		gt_bounds = gt.squeeze(0)[:,0].min().item(), gt.squeeze(0)[:,1].min().item(), gt.squeeze(0)[:,0].max().item(), gt.squeeze(0)[:,1].max().item()

		bounds = [(pred_bounds[b], gt_bounds[b]) for b in range(len(pred_bounds))]
		bounds[0] = min(bounds[0])
		bounds[1] = min(bounds[1])
		bounds[2] = max(bounds[2])
		bounds[3] = max(bounds[3])
		
		# Line should fall within bounds of pred polygons
		# if not ((points[0] > bounds[0] and points[1] > bounds[1] and points[0] < bounds[2] and points[1] < bounds[3]) and ((points[2] > bounds[0] and points[3] > bounds[1] and points[2] < bounds[2] and points[3] < bounds[3]))):
		# 	reward += -10
		if not ((points[0] > bounds[0] and points[0] < bounds[2]) and ((points[2] > bounds[0] and points[2] < bounds[2]))):
			reward += -10

		if not gt_intersect:
			if self.flags['num_intersections'] == 0:
				reward += 1 # Maybe this should be 0?
			elif self.flags['diff_polygons']:
				reward += 2
			elif (self.flags['num_intersections'] != 2) or (self.flags['degenerate_split']):
				reward += 6
			else:
				reward += 10
				# print ("Good Split")
		else:
			if self.flags['num_intersections'] == 0:
				reward += 0
			elif self.flags['diff_polygons']:
				reward += 1
			elif (self.flags['num_intersections'] != 2) or (self.flags['degenerate_split']):
				reward += 3
			else:
				reward += 5
				# print ("OK Split")
		add_loss = not gt_intersect
		return data, reward, done, add_loss


	def get_intersection(self, c, edge_index, points):

		edge_index_l, edge_index_r = edge_index

		x1, y1 = c[edge_index_l].transpose(0,1)
		x2, y2 = c[edge_index_r].transpose(0,1)

		[p1, q1, p2, q2] = points.tolist()

		intersect_mask = (self.line(p1,q1,p2,q2,x1,y1)*self.line(p1,q1,p2,q2,x2,y2) < 0) & (self.line(x1,y1,x2,y2,p1,q1)*self.line(x1,y1,x2,y2,p2,q2) < 0) & (self.line(p1,q1,p2,q2,x1,y1) > 0)
		if intersect_mask.sum().item() != 2:
			return None, None, intersect_mask.sum().item()
		else:
			return edge_index_l[intersect_mask], edge_index_r[intersect_mask], 2

	def line(self, p1,q1,p2,q2,x1,y1):
			return (p2-p1)*y1 - (q2-q1)*x1 -(q1*p2-q2*p1)






		