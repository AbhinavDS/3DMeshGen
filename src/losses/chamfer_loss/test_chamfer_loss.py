import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from chamfer_loss import ChamferLoss

chamfer = ChamferLoss()