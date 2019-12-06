"""
Module that controls the training of the graphQA module
"""

import os
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
from  src import dtypeF, dtypeL, dtypeB

class Trainer:

	def __init__(self, params, train_generator, val_generator):

		self.params = params
		self.num_epochs = params.num_epochs
		self.train_generator = train_generator
		self.val_generator = val_generator
		# self.model = Model(params)
		self.model = nn.Linear(10,100)
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

			lr = self.adjust_lr(epoch)
			self.model.train()

			loss = 0.0
			train_accuracies = []

			for i, range(int(math.ceil(self.params.data_size/self.params.batch_size))):

				self.optimizer.zero_grad()

				gt_vertices, gt_normals, edges, gt_image_feats, proj_gt = next(self.train_generator)

				gt_vertices = torch.Tensor(gt_vertices).type(dtypeF).requires_grad_(False)
				gt_normals = torch.Tensor(gt_normals).type(dtypeF).requires_grad_(False)

				#Need some padding conversion
				gt_image_feats = torch.Tensor(gt_image_feats).type(dtypeF).requires_grad_(False)

				mask = None
				self.model.forward(gt_image_feats, gt_vertices, gtnormals, mask)

				total_closs += self.model.closs/len(train_data)
				total_laploss += self.model.laploss/len(train_data)
				total_nloss += self.model.nloss/len(train_data)
				total_eloss += self.model.eloss/len(train_data)
				total_loss += self.model.loss/len(train_data)
			
				self.model.loss.backward()

				nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
				
				self.optimizer.step()
				
				if i % self.params.display_every == 0:
					print('Train Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, model.loss))

			train_acc = np.mean(train_accuracies)
			self.save_ckpt(save_best=False)
			self.write_status(epoch, self.model.loss)
			
	
	
	def check_restart_conditions(self):

		# Check for the status file corresponding to the model
		status_file = os.path.join(self.params.ckpt_dir, 'status.json')

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

