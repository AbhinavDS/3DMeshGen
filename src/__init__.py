import torch
if torch.cuda.is_available():
	dtypeF = torch.cuda.FloatTensor
	dtypeL = torch.cuda.LongTensor
	dtypeB = torch.cuda.ByteTensor
else:
	dtypeF = torch.FloatTensor
	dtypeL = torch.LongTensor
	dtypeB = torch.ByteTensor
