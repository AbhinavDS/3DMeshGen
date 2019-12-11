import datetime
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import itertools
import torch
from gym import spaces
from tensorboardX import SummaryWriter
from torch_geometric.utils import to_dense_adj
from src.modules.splitter_rl.splitter import Splitter
from src.util import utils
from src.modules.splitter_rl.sac.sac import SAC
from src.modules.splitter_rl.sac.replay_memory import ReplayMemory
from src import PAD_TOKEN

class RLAgent:
	def __init__(self, params, max_steps, agent='sac', seeded=True):
		self.params = params
		self.max_steps = max_steps
		

		# Memory
		self.memory = ReplayMemory(params.replay_size)

		# Splitter
		self.splitter = Splitter(params)


		#TensorboardX
		self.writer = SummaryWriter(logdir=f'{self.params.log_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{params.rl_policy} {"_autotune" if params.automatic_entropy_tuning else ""}')

		self.updates = 0
		
		self.losses = None
		self.action_dim = 2*self.params.dim_size + 1
		self.max_abs_action = 1
		self.total_numsteps = 0
		if seeded:
			self.seed()
		
		# Agent
		self.action_space = spaces.Box( -1*np.ones(self.action_dim)*(self.max_abs_action), np.ones(self.action_dim)*(self.max_abs_action))
		# self.observation_space_size = self.params.img_width*2 + self.params.feature_size
		# self.observation_space_size = self.params.img_width + 2*(self.params.max_total_vertices+self.params.max_polygons)
		# self.observation_space_size = self.params.img_width//2 + 2*(self.params.max_total_vertices+self.params.max_polygons)
		self.observation_space_size = self.params.img_width*2
		print (self.observation_space_size)
		self.agent = None
		if agent == 'sac':
			self.agent = SAC(self.observation_space_size, self.action_space, params)

	def create_state(self, data, image_features, proj_gt):
		x, c, pid = data
		proj_vertices = utils.scaleBackTensor(c.x).detach().cpu().numpy().astype(np.int)
		proj_adj_mat = to_dense_adj(c.edge_index).squeeze(0).detach().cpu().numpy().astype(np.int)
		proj_pred = utils.flatten_pred(proj_vertices, proj_adj_mat, self.params)
		proj_gt = np.squeeze(proj_gt, axis = 0)
		normalized_img_features = (image_features.cpu().numpy().squeeze(0) - PAD_TOKEN)/(self.params.img_width - PAD_TOKEN)
		#state = np.concatenate((proj_gt, proj_pred, normalized_img_features))
		state = np.concatenate((proj_gt, proj_pred))
		# state = np.concatenate((proj_gt[::2], proj_pred[::2], normalized_img_features[:2*(self.params.max_total_vertices+self.params.max_polygons)]))
		#state = np.concatenate(((2*proj_gt-proj_pred)[::2], normalized_img_features[:2*(self.params.max_total_vertices+self.params.max_polygons)]))
		#state = 2 * state - 1
		return state, proj_pred
		
	def seed(self):
		torch.manual_seed(self.params.seed)
		np.random.seed(self.params.seed)
		
	def train(self, deformer_block, init_data, image_features, gt, gt_normals, proj_gt, gt_edges, gt_num_polygons):
		
		count_add_loss = 0.0
		i_episode = 0
		data = init_data
		for i_episode in range(self.params.rl_num_episodes):
			episode_reward = 0
			episode_steps = 0
			done = False
			state, proj_pred = self.create_state(init_data, image_features, proj_gt)
			data = (init_data[0].clone(), init_data[1].clone(), init_data[2].clone())
			while not done:
				# print (f'Episode Step {episode_steps}')
				if self.params.start_steps > self.total_numsteps:
					action = self.action_space.sample() # Sample random action
				else:
					action = self.agent.select_action(state)  # Sample action from policy
				if len(self.memory) > self.params.rl_batch_size:
					# Number of updates per step in environment
					for i in range(self.params.updates_per_step):
						# Update parameters of all the networks
						critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.memory, self.params.rl_batch_size, self.updates)
						#print(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha)
						self.writer.add_scalar('loss/critic_1', critic_1_loss, self.updates)
						self.writer.add_scalar('loss/critic_2', critic_2_loss, self.updates)
						self.writer.add_scalar('loss/policy', policy_loss, self.updates)
						self.writer.add_scalar('loss/entropy_loss', ent_loss, self.updates)
						self.writer.add_scalar('entropy_temprature/alpha', alpha, self.updates)
						self.updates += 1

				data, reward, done, add_loss = self.splitter.split_and_reward(data, action, gt, gt_edges, gt_num_polygons)
				deformer_block.set_loss_to_zero()
				# add_loss = False
				data = deformer_block.forward(data[0], data[1], image_features, data[2], gt, gt_normals, add_loss = add_loss)
				if add_loss:
					count_add_loss += 1.0
					# if deformer_block.projection.W_p.weight.grad is not None:
					# 	print ("Before","db2.projection.W_p.weight", torch.max(deformer_block.projection.W_p.weight.grad))
					deformer_block.loss.backward(retain_graph=True)
					# print ("After","db2.projection.W_p.weight", torch.max(deformer_block.projection.W_p.weight.grad))
					# print ("Seems to be working") # 
				
				next_state, proj_pred = self.create_state(data, image_features, proj_gt)
				episode_steps += 1
				self.total_numsteps += 1
				episode_reward += reward

				# Ignore the "done" signal if it comes from hitting the time horizon.
				# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
				


				done = True
				





				mask = 0 if episode_steps == self.max_steps else float(done)

				self.memory.push(state, action, reward, next_state, mask) # Append transition to memory

				state = next_state

				# Terminate episode forcefully
				done = done or (episode_steps >= self.max_steps)
				break

			self.writer.add_scalar('reward/train', episode_reward, i_episode)
			if i_episode % self.params.display_every == 0:
				print("RL Episode: {}, episode steps: {}, reward: {}".format(i_episode, episode_steps, round(episode_reward, 2)))
		
		# utils.drawPolygons(utils.scaleBack(data[1].x), utils.scaleBack(gt[0]), gt_edges[0], proj_pred=state[self.params.img_width:2*self.params.img_width], proj_gt=state[:self.params.img_width], color='red',out=self.params.expt_res_dir+f'/../train_out_rl{episode_steps}.png',A=to_dense_adj(data[1].edge_index).cpu().numpy()[0], line=action[:4])
							
		if self.params.rl_eval == True and self.params.rl_num_episodes:
			avg_reward = 0.
			episodes = 1
			for _  in range(episodes):
				state, proj_pred = self.create_state(init_data, image_features, proj_gt)
				data = (init_data[0].clone(), init_data[1].clone(), init_data[2].clone())
				episode_reward = 0
				done = False
				episode_steps = 0
				reward = 0
				while not done:
					print (f'Episode Step {episode_steps}; Done {done}; Reward; {reward}')
					action = self.agent.select_action(state, eval=True)

					data, reward, done, _ = self.splitter.split_and_reward(data, action, gt, gt_edges, gt_num_polygons)

					utils.drawPolygons(utils.scaleBack(data[1].x), utils.scaleBack(gt[0]), gt_edges[0], proj_pred=proj_pred, proj_gt=proj_gt[0], color='red',out=self.params.expt_res_dir+f'/../train_out_rl{episode_steps}.png',A=to_dense_adj(data[1].edge_index).cpu().numpy()[0], line=action[:4], text=f'Reward {reward}, Done {action[4]:4f}, Random: {self.params.start_steps > self.total_numsteps}')

					data = deformer_block.forward(data[0], data[1], image_features, data[2], gt, gt_normals, add_loss = False)

					episode_reward += reward
					episode_steps += 1
					state, proj_pred = self.create_state(data, image_features, proj_gt)
					
					# Terminate episode forcefully
					done = done or (episode_steps >= self.max_steps)
					break

				avg_reward += episode_reward
			avg_reward /= episodes


			self.writer.add_scalar('avg_reward/test', avg_reward, i_episode)

			print("----------------------------------------")
			print("Test Episodes: {}, Return: {}".format(episodes, round(avg_reward, 2)))
			print("----------------------------------------")

		# Scale all the losses or keep number of episodes less
		# if count_add_loss:
		# 	deformer_block.scaleLosses(1.0/count_add_loss)
		return data[1]