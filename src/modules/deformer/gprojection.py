import torch.nn as nn
import torch.nn.functional as F
class GProjection(nn.Module):

    def __init__(self, feature_size, dim_size, weights_init = 'zero'):
        super(GProjection, self).__init__()

        self.W_p_c = nn.Linear(self.dim_size, self.feature_size)
        self.W_p_s = nn.Linear(self.feature_size, self.feature_size)
        self.W_p = nn.Linear(2*self.feature_size,self.feature_size)
        self.activation = F.relu
        #self.activation = nn.Tanh()
        self.dropout = nn.Dropout()
        
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
        c_f = self.dropout(self.activation(self.W_p_c(c)))
        s_f = self.dropout(self.activation(self.W_p_s(s)))
        feature_from_state = self.W_p(torch.cat((c_f,s_f),dim=2))
        return feature_from_state