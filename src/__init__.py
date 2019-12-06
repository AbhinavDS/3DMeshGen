import torch

MEAN = 300
VAR = 300
PAD_TOKEN = -2
IMG_PAD_TOKEN = -20

if torch.cuda.is_available():
	dtypeF = torch.cuda.FloatTensor
	dtypeL = torch.cuda.LongTensor
	dtypeB = torch.cuda.ByteTensor
else:
	dtypeF = torch.FloatTensor
	dtypeL = torch.LongTensor
	dtypeB = torch.ByteTensor
