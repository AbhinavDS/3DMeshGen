import torch
import torch.nn as nn
class GProjection(nn.Module):

	def __init__(self, feature_size, dim_size, weights_init = 'normal'):
		super(GProjection, self).__init__()

		self.W_p_c = nn.Linear(dim_size, feature_size)
		self.W_p_s = nn.Linear(feature_size, feature_size)
		self.W_p = nn.Linear(2*feature_size, feature_size)
		#self.activation = nn.ReLU()
		self.activation = nn.Tanh()
		# self.dropout = nn.Dropout()
		self.dropout = nn.Dropout(p=0)
		
		if weights_init == 'zero':
			self.zero_init()
		else:
			self.xavier_init()
	def xavier_init(self):
		nn.init.xavier_uniform_(self.W_p_c.weight)
		nn.init.xavier_uniform_(self.W_p_s.weight)
		nn.init.xavier_uniform_(self.W_p.weight)

	def zero_init(self):
		nn.init.constant_(self.W_p_c.weight,0)
		nn.init.constant_(self.W_p_s.weight,0)
		nn.init.constant_(self.W_p.weight,0)
	
	def forward(self, c, s):
		#c: 1 x N x d
		#s: 1 x f
		c = c.unsqueeze(0)
		c_f = self.dropout(self.activation(self.W_p_c(c)))
		s_f = self.dropout(self.activation(self.W_p_s(s))).unsqueeze(1).expand(-1,c_f.size(1),-1)
		feature_from_state = self.W_p(torch.cat((c_f,s_f),dim=2))
		return feature_from_state.squeeze(0)