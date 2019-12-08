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
from models.pixel2mesh import Pixel2Mesh as Model

from src.losses.chamfer_loss.chamfer_loss import ChamferLoss
from src.losses.edge_loss.edge_loss import EdgeLoss
from src.losses.normal_loss.normal_loss import NormalLoss

class Trainer:

	def __init__(self, params, train_generator, val_generator):

		self.params = params
		self.num_epochs = params.num_epochs
		self.train_generator = train_generator
		self.val_generator = val_generator
		self.model = Model(params)
		self.device = self.params.device

		# Can be changed to support different optimizers
		# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
		self.optimizer = torch.optim.Adamax(self.model.parameters())
		self.lr = self.params.lr
		self.log = params.log
		if self.log:
			self.writer = SummaryWriter(params.log_dir)

	def train(self):

		print('Initiating Training')

		# Check if the training has to be restarted
		self.check_restart_conditions()
		
		if self.resume_from_epoch >= 1:
			print('Loading from Checkpointed Model')
			# Write the logic for loading checkpoint for the model
			self.load_ckpt()

		self.model.to(self.device)
		
		for epoch in range(self.resume_from_epoch, self.num_epochs):

			total_closs = 0.0
			total_laploss = 0.0
			total_nloss = 0.0
			total_eloss = 0.0
			total_loss = 0.0

			num_iters = int(math.ceil(self.params.data_size/self.params.batch_size))
			lr = self.adjust_lr(epoch)
			for g in self.optimizer.param_groups:
				g['lr'] = lr
			self.model.train()

			loss = 0.0
			train_accuracies = []

			for i in range(num_iters):

				self.optimizer.zero_grad()

				gt_vertices, gt_normals, gt_edges, gt_image_feats, proj_gt = next(self.train_generator)
				
				gt_vertices = torch.Tensor(gt_vertices).type(dtypeF).requires_grad_(False)
				gt_normals = torch.Tensor(gt_normals).type(dtypeF).requires_grad_(False)
				gt_image_feats = torch.Tensor(gt_image_feats).type(dtypeF).requires_grad_(False)

				x, c = self.model.forward(gt_image_feats, gt_vertices, gt_normals)

				total_closs += self.model.closs/num_iters
				total_laploss += self.model.laploss/num_iters
				total_nloss += self.model.nloss/num_iters
				total_eloss += self.model.eloss/num_iters
				total_loss += self.model.loss/num_iters
			
				self.model.loss.backward()

				# nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
				
				self.optimizer.step()
				
				if i % self.params.display_every == 0:
					print(f'Train Epoch: {epoch}, Iteration: {i}, LR: {lr}, Loss: {self.model.loss}, CLoss: {self.model.closs}, NLoss: {self.model.nloss}, ELoss: {self.model.eloss}, LapLoss: {self.model.laploss}')
					# proj_pred = utils.flatten_pred_batch(utils.scaleBack(c.x), A, self.params)
					utils.drawPolygons(utils.scaleBack(c.x), utils.scaleBack(gt_vertices[0]), gt_edges[0], proj_pred=None, proj_gt=None, color='red',out=self.params.expt_res_dir+'/../out.png',A=to_dense_adj(c.edge_index).cpu().numpy()[0])
				

			train_acc = np.mean(train_accuracies)
			self.save_ckpt(save_best=False)
			self.write_status(epoch, total_loss.item())
			
	
	
	def check_restart_conditions(self):

		# Check for the status file corresponding to the model
		status_file = os.path.join(self.params.ckpt_dir, 'status.json')
		print (status_file)
		if os.path.exists(status_file):
			with open(status_file, 'r') as f:
				status = json.load(f)			
			self.resume_from_epoch = status['epoch']
			self.best_val_acc = status['best_val_acc']
		else:
			self.resume_from_epoch = 0
			self.best_val_acc = 0.0

	def write_status(self, epoch, best_val_acc):

		status_file = os.path.join(self.params.ckpt_dir, 'status.json')
		status = {'epoch': epoch, 'best_val_acc': best_val_acc}

		with open(status_file, 'w') as f:
			json.dump(status, f, indent=4)

	def adjust_lr(self, epoch):
		
		# Sets the learning rate to the initial LR decayed by 2 every learning_rate_decay_every epochs
		
		lr_tmp = self.lr * (0.5 ** (epoch // self.params.learning_rate_decay_every))
		return lr_tmp

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

	def save_ckpt(self, save_best=False):

		"""
		Saves the model checkpoint at the correct directory path
		"""

		model_name = self.model.__class__.__name__
		ckpt_path = os.path.join(self.params.ckpt_dir, '{}.ckpt'.format(model_name))

		# Maybe add more information to the checkpoint
		model_dict = {
			'state_dict': self.model.state_dict(),
			'params': self.params
		}

		torch.save(model_dict, ckpt_path)

		if save_best:
			best_ckpt_path = os.path.join(self.params.ckpt_dir, '{}_best.ckpt'.format(model_name))
			torch.save(model_dict, best_ckpt_path)

