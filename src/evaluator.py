"""
Module that controls the training of the graphQA module
"""

import os
import math
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
from torch_geometric.utils import to_dense_adj
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from src import dtypeF, dtypeL, dtypeB
from src.util import utils
from src.models.pixel2mesh import Pixel2Mesh
from src.models.pixel2mesh_rl import Pixel2MeshRL

class Evaluator:

	def __init__(self, params, test_generator):

		self.params = params
		self.test_generator = test_generator
		if self.params.rl_model:
			self.model = Pixel2MeshRL(params)
		else:
			self.model = Pixel2Mesh(params)
		self.device = self.params.device

		self.log = params.log
		if self.log:
			self.writer = SummaryWriter(params.log_dir)

	def eval(self):

		print('Initiating Evaluation')
		
		print('Loading from Checkpointed Model')
		self.load_ckpt()

		self.model.to(self.device)
		
		total_closs = 0.0
		total_laploss = 0.0
		total_nloss = 0.0
		total_eloss = 0.0
		total_loss = 0.0

		num_iters = int(math.ceil(self.params.test_data_size/self.params.batch_size))
		self.model.eval()

		loss = 0.0
		
		for i in range(num_iters):
			gt_vertices, gt_normals, gt_edges, gt_image_feats, proj_gt = next(self.test_generator)
			
			gt_vertices = torch.Tensor(gt_vertices).type(dtypeF).requires_grad_(False)
			gt_normals = torch.Tensor(gt_normals).type(dtypeF).requires_grad_(False)
			gt_image_feats = torch.Tensor(gt_image_feats).type(dtypeF).requires_grad_(False)

			c = self.model.forward(gt_image_feats, gt_vertices, gt_normals, proj_gt)

			total_closs += self.model.closs/num_iters
			total_laploss += self.model.laploss/num_iters
			total_nloss += self.model.nloss/num_iters
			total_eloss += self.model.eloss/num_iters
			total_loss += self.model.loss/num_iters
		
		print(f'Evaluator: Loss: {total_loss:.4f}, CLoss: {total_closs:.4f}, NLoss: {total_nloss:.4f}, ELoss: {total_eloss:.4f}, LapLoss: {total_laploss:.4f}')
		# proj_pred = utils.flatten_pred_batch(utils.scaleBack(c.x), A, self.params)
		utils.drawPolygons(utils.scaleBack(c.x), utils.scaleBack(gt_vertices[0]), gt_edges[0], proj_pred=None, proj_gt=None, color='red',out=self.params.expt_res_dir+'/../test_out.png',A=to_dense_adj(c.edge_index).cpu().numpy()[0])
		
		self.write_status(total_loss.item())
	

	def write_status(self, best_val_acc):

		status_file = os.path.join(self.params.ckpt_dir, 'test_status.json')
		status = {'best_val_acc': best_val_acc}

		with open(status_file, 'w') as f:
			json.dump(status, f, indent=4)


	def log_stats(self, train_loss, val_loss, train_acc, val_acc, epoch):
		
		"""
		Log the stats of the current
		"""

		if self.log:
			self.writer.add_scalar('train/loss', train_loss, epoch)
			self.writer.add_scalar('train/acc', train_acc, epoch)
			self.writer.add_scalar('val/loss', val_loss, epoch)
			self.writer.add_scalar('val/acc', val_acc, epoch)

	def load_ckpt(self):
		"""
		Load the model checkpoint from the provided path
		"""

		# TODO: Maybe load params as well from the checkpoint

		model_name = self.model.__class__.__name__
		ckpt_path = os.path.join(self.params.ckpt_dir, '{}.ckpt'.format(model_name))

		ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		self.model.load_state_dict(ckpt['state_dict'])
		self.model.load_model(suffix=model_name)