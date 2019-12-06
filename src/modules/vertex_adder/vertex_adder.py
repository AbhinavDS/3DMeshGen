import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB
from torch_geometric.utils import to_undirected

class VertexAdder(nn.Module):
	def __init__(self):
		super(VertexAdder, self).__init__()
		pass

	def forward(self, x_prev, c_prev, edge_index, Pid):#Pid update pending
		"""
		x_prev: 1 x v x f
		c_prev: 1 x v x d
		edge_index: 1 x 2 x e
		"""
		_,_,num_edges = edge_index.size()
		_,num_nodes,feature_size = x_prev.size()
		new_num_nodes = num_edges + num_nodes
		
		d_mask = (edge_index[:,0] < edge_index[:,1])
		d_edge_index = torch.masked_select(edge_index, d_mask).reshape(edge_index.size(0), 2, -1).type_as(edge_index)
		d_edge_index_l, d_edge_index_r = d_edge_index[0]
		
		new_edge_index =  torch.Tensor(edge_index.size(0), edge_index.size(1), edge_index.size(2)*2).type_as(edge_index)
		new_edge_index[0,0,:num_edges] = d_edge_index_l
		new_edge_index[0,1,:num_edges] = torch.arange(num_nodes, new_num_nodes).type_as(edge_index)
		new_edge_index[0,0,num_edges:] = torch.arange(num_nodes, new_num_nodes).type_as(edge_index)
		new_edge_index[0,1,num_edges:] = d_edge_index_r
		new_edge_index = to_undirected(new_edge_index, new_num_nodes)

		new_x = torch.zeros(x.size(0), new_num_nodes, x.size(2)).type_as(x_prev)
		new_c = torch.zeros(c.size(0), new_num_nodes, c.size(2)).type_as(c_prev)

		new_x[:,:num_nodes] = x_prev
		new_c[:,:num_nodes] = c_prev

		new_x[:,num_nodes:] = 0.5*(x_prev[:,d_edge_index_l] + x_prev[:,d_edge_index_r])
		new_c[:,num_nodes:] = 0.5*(c_prev[:,d_edge_indec_l] + c_prev[:,d_edge_index_r])

		return new_x, new_c, new_edge_index, Pid

	def forward(self, x_prev, c_prev, A, Pid, s_prev):
		# dim A: batch_size x vertices x vertices
		batch_size = A.shape[0]
		feature_size = x_prev.size(2)
		dim_size = c_prev.size(2)
		num_vertices = A.shape[1] * np.ones(batch_size)
		Ar = np.reshape(A, (batch_size, -1))
		final_num_vertices = num_vertices +  np.count_nonzero(Ar, axis=1)/2

		num_vertices = int(num_vertices[0])
		final_num_vertices = int(final_num_vertices[0])
		A_new = np.zeros((batch_size, final_num_vertices, final_num_vertices))
		Pid_new = np.zeros((batch_size, final_num_vertices, final_num_vertices))

		#v_index: batch_size
		v_index = np.ones(batch_size,dtype=np.int)*num_vertices#first new vertex added here
		v_index = np.expand_dims(v_index, 1)
		#x_prev: batch x num_vert x feat 
		x_new =  torch.cat((x_prev,torch.zeros(batch_size,final_num_vertices-num_vertices,feature_size).type(dtype)),dim=1)
		c_new =  torch.cat((c_prev,torch.zeros(batch_size,final_num_vertices-num_vertices,dim_size).type(dtype)),dim=1)
		s_new =  torch.cat((s_prev,torch.zeros(batch_size,final_num_vertices-num_vertices,feature_size).type(dtype)),dim=1)
		for i in range(num_vertices):
			# k = i+1
			polygon_i = np.max(Pid[:,:,i], axis=-1)
			polygon_i = np.expand_dims(polygon_i, 1)

			for j in range(i+1, num_vertices):				
				# k += 1
			
				mask = np.expand_dims(A[:,i,j],1)
				pmask = polygon_i * mask
				#mask: batch_size
				temp_A_new = np.zeros((batch_size, final_num_vertices, final_num_vertices))
				
				#add vertex between them
				np.put_along_axis(temp_A_new[:,i,:], v_index, mask, axis=1)
				np.put_along_axis(temp_A_new[:,:,i], v_index, mask, axis=1)
				np.put_along_axis(temp_A_new[:,:,j], v_index, mask, axis=1)
				np.put_along_axis(temp_A_new[:,j,:], v_index, mask, axis=1)

				A_new += temp_A_new


				# Do same for Pid
				temp_Pid_new = np.zeros((batch_size, final_num_vertices, final_num_vertices))
				
				np.put_along_axis(temp_Pid_new[:,i,:], v_index, pmask, axis=1)
				np.put_along_axis(temp_Pid_new[:,:,i], v_index, pmask, axis=1)
				np.put_along_axis(temp_Pid_new[:,:,j], v_index, pmask, axis=1)
				np.put_along_axis(temp_Pid_new[:,j,:], v_index, pmask, axis=1)

				Pid_new += temp_Pid_new

				#x_new : batch x final_num_vert x feat
				tmask = torch.LongTensor(mask).type(dtype)
				tv_index = torch.LongTensor(v_index).type(dtypeL)			
				x_v = ((x_prev[:,i,:] + x_prev[:,j,:])/2)*tmask#batch x feat
				c_v = ((c_prev[:,i,:] + c_prev[:,j,:])/2)*tmask
				s_v = ((s_prev[:,i,:] + s_prev[:,j,:])/2)*tmask
				x_v = x_v.unsqueeze(1)
				c_v = c_v.unsqueeze(1)
				s_v = s_v.unsqueeze(1)
				tv_index = tv_index.unsqueeze(1)
				x_new.scatter_add_(1,tv_index.repeat(1, 1, feature_size), x_v)
				c_new.scatter_add_(1,tv_index.repeat(1, 1, dim_size), c_v)
				s_new.scatter_add_(1,tv_index.repeat(1, 1, feature_size), s_v)
				v_index += mask.astype(int)
				v_index = v_index % final_num_vertices
		return x_new, c_new, A_new, Pid_new, s_new