import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
from edge_loss import EdgeLoss
def verif_edge(pred, A):
	temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
	
	# Might not make sense in 2D
	edges = regularizer(pred, temp_A)
	loss = torch.mul(edges, edges)
	loss = torch.sum(loss, dim =1)
	return torch.mean(loss[(temp_A != 0)])

def regularizer(x, A):
	batch_size, num_points_x, points_dim = x.size()
	x = x.permute(0,2,1)
	x = x.repeat(1,1,num_points_x).view(batch_size, points_dim, num_points_x, num_points_x)
	x_t = x.transpose(3,2)
	x_diff = x_t - x
	
	# Filter out non-neighbours		
	A = A.unsqueeze(1)
	A = A.repeat(1,points_dim,1,1)
	x_diff = torch.mul(x_diff, A)
	return x_diff

eloss = EdgeLoss()

num_points = 10
num_gt_points = 20
points_dim = 3
batch_size = 2

preds = torch.rand(batch_size, num_points, points_dim)
A = np.zeros((batch_size, num_points, num_points))
edge_list = np.zeros((batch_size, 2, 2*num_points*num_points))
for b in range(batch_size):
	counter = 0
	for i in range(num_points):
		for j in range(i+1, num_points):
				if random.randint(0,1):
					A[b,i,j] = 1
					A[b,j,i] = 1
					edge_list[b,0,counter] = i
					edge_list[b,1,counter] = j
					counter += 1
					edge_list[b,0,counter] = j
					edge_list[b,1,counter] = i
					counter += 1
edge_list = torch.from_numpy(edge_list).long()

print(eloss.forward(preds, edge_list))
print(verif_edge(preds, A))