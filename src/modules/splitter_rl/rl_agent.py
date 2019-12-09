import datetime
import numpy as np
import itertools
import torch
from tensorboardX import SummaryWriter
from torch_geometric.utils import to_dense_adj
from src.util import utils
from sac.sac import SAC
from sac.replay_memory import ReplayMemory


class RLAgent:
	def __init__(self, params, max_steps, agent='sac', seeded=True):
		self.params = params
		self.max_steps = max_steps
		
		# Agent
		self.agent = None
		if agent == 'sac':
			#TODO: CJANMGE
			self.agent = SAC(env.observation_space.shape[0], env.action_space, params)

		# Memory
		self.memory = ReplayMemory(params.replay_size)


		#TensorboardX
		self.writer = SummaryWriter(logdir='{}/{}_SAC_{}_{}_{}'.format(self.params.log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), params.rl_policy, "autotune" if params.automatic_entropy_tuning else ""))

		self.updates = 0
		
		self.losses = None
		self.action_dim = 2*self.params.dim_size + 1
		self.max_abs_action = 1
		self.total_numsteps = 0
		if seeded:
			self.seed()

	def random_action(): #Define action space
		return np.random.rand(5)*self.max_abs_action
		
	def create_state(self, data, image_features, proj_gt):
		x, c, pid = data
		proj_pred = utils.flatten_pred(utils.scaleBack(c.x), to_dense_adj(c.edge_index), self.params)
		proj_gt = np.squeeze(proj_gt, axis = 0)
		x, c, pid = x.x.cpu().numpy().flatten(), c.x.cpu().numpy().flatten(), pid.x.cpu().numpy().flatten() 
		shadow = (2*proj_gt - proj_pred)[::2]
		state = np.concatenate((x, c, pid, image_features, shadow))
		return state

	def seed(self):
		torch.manual_seed(self.params.seed)
		np.random.seed(self.params.seed)
		
	def train(self, deformer_block, init_data, image_features, gt, gt_normals, proj_gt, gt_edges, gt_num_polygons):
		
		for i_episode in range(self.params.rl_num_episodes):
			episode_reward = 0
			episode_steps = 0
			done = False
			state = self.create_state(init_data, image_features, proj_gt)
			data = init_data
			
			deformer_block.set_loss_to_zero()
			while not done:
				if self.params.start_steps > self.total_numsteps:
					action = self.random_action()  # Sample random action
				else:
					action = self.agent.select_action(state)  # Sample action from policy
				if len(self.memory) > self.params.batch_size:
					# Number of updates per step in environment
					for i in range(self.params.updates_per_step):
						# Update parameters of all the networks
						critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.memory, self.params.batch_size, self.updates)

						self.writer.add_scalar('loss/critic_1', critic_1_loss, self.updates)
						self.writer.add_scalar('loss/critic_2', critic_2_loss, self.updates)
						self.writer.add_scalar('loss/policy', policy_loss, self.updates)
						self.writer.add_scalar('loss/entropy_loss', ent_loss, self.updates)
						self.writer.add_scalar('entropy_temprature/alpha', alpha, self.updates)
						self.updates += 1

				data, reward, done, add_loss = self.splitter.split_and_reward(data, action, gt, gt_edges, gt_num_polygons)
				data = deformer_block.forward(data[0], data[1], image_features, data[2], gt, gt_normals, add_loss = add_loss)
				
				next_state = self.create_state(data, image_features, proj_gt)
				episode_steps += 1
				self.total_numsteps += 1
				episode_reward += reward

				# Ignore the "done" signal if it comes from hitting the time horizon.
				# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
				mask = 0 if episode_steps == env._max_episode_steps else float(done)

				self.memory.push(state, action, reward, next_state, mask) # Append transition to memory

				state = next_state

				# Terminate episode forcefully
				done = (episode_steps >= self.max_steps)

			writer.add_scalar('reward/train', episode_reward, i_episode)
			print("Episode: {}, episode steps: {}, reward: {}".format(i_episode, episode_steps, round(episode_reward, 2)))

		if i_episode % 10 == 0 and self.params.rl_eval == True:
			avg_reward = 0.
			episodes = 10
			for _  in range(episodes):
				state = self.create_state(init_data, image_features, proj_gt)
				episode_reward = 0
				done = False
				while not done:
					action = self.agent.select_action(state, eval=True)

					data, reward, done, _ = self.splitter.split_and_reward(data, action, gt, gt_edges, gt_num_polygons)
					data = deformer_block.forward(data[0], data[1], image_features, data[2], gt, gt_normals, add_loss = False)
					
					episode_reward += reward
					state = self.create_state(data, image_features, proj_gt)

				avg_reward += episode_reward
			avg_reward /= episodes


			self.writer.add_scalar('avg_reward/test', avg_reward, i_episode)

			print("----------------------------------------")
			print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
			print("----------------------------------------")


