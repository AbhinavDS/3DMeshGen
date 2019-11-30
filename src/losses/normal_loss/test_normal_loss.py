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
from normal_loss import NormalLoss

nloss = NormalLoss()

num_points = 100
num_gt_points = 200
points_dim = 3
batch_size = 16

preds = torch.rand(batch_size, num_points, points_dim)
gt_normals = torch.rand(batch_size, num_gt_points, points_dim)
A = np.zeros((batch_size, num_points, num_points))
edge_list = np.zeros((batch_size, 2, 2*num_points*num_points))
nearest_gt = []
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

	nearest_gt.append([random.randint(0, num_gt_points - 1) for p in range(0, num_points)])
edge_list = torch.from_numpy(edge_list).long()
nearest_gt = torch.from_numpy(np.array(nearest_gt))

# xp = [preds[0][i][0] for i in range(num_points)] + [preds[0][0][0]]
# yp = [preds[0][i][1] for i in range(num_points)] + [preds[0][0][1]]
# xg = [preds[0][i][0] for i in range(num_points)] + [preds[0][0][0]]
# yg = [preds[0][i][1] for i in range(num_points)] + [preds[0][0][1]]
# plt.subplot(3,1,1)
# plt.plot(xp, yp, 'go-', label='line 1', linewidth=2)
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(xg, yg, 'ro-', label='line 1', linewidth=2)
# plt.legend()
# plt.show()

def normal_pred(x, A):
	batch_size, num_points_x, points_dim = x.size()
	x = x.permute(0,2,1)
	x = x.unsqueeze(-1).expand(-1,-1,-1,num_points_x)
	x_t = x.transpose(3,2)
	x_diff = x_t - x
	x_diff = F.normalize(x_diff, dim = 1)

	# Filter out non-neighbours		
	A = A.unsqueeze(1).expand(-1, points_dim, -1, -1)
	x_diff = torch.mul(x_diff, A)
	return x_diff

def calculate_loss(p_k, nq, mask):
	batch_size, points_dim, num_points_x, _ = p_k.size()
	nq = nq.transpose(2,1)
	nq = nq.unsqueeze(2).expand(-1,-1,-1,num_points_x)
	inner_product = torch.mul(nq, p_k)
	
	inner_product = torch.sum(inner_product, dim=1)
	inner_product = torch.mul(inner_product,inner_product)
	return torch.mean(inner_product[mask])

def verif_forward(preds, nearest_gt, gt_normals, A):
	temp_A = Variable(torch.Tensor(A).type(dtype), requires_grad=False)
	q = F.normalize(torch.gather(gt_normals, 1, nearest_gt.unsqueeze(2).expand(-1,-1,points_dim)), dim = 2)

	# Calculate difference for each pred vertex, use adj mat to filter out non-neighbours
	diff_neighbours = normal_pred(preds, temp_A)
	# Calculate final loss
	return calculate_loss(diff_neighbours, q, (temp_A != 0))


print(nloss.forward(preds, nearest_gt, gt_normals, edge_list))
print(verif_forward(preds, nearest_gt, gt_normals, A))

