"""
Module that parses the command line arguments
"""

import argparse
import os
import torch

class Config(object):

	def gen_config(self):
		
		"""
		Function to be invoked after arguments have been processed to generate additional config variables
		"""

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print('Executing on Device: {}'.format(self.device))
		
		self.expt_res_dir = os.path.join(self.expt_res_base_dir, self.expt_name)
		self.log_dir = os.path.join(self.expt_res_dir, 'logs')
		self.ckpt_dir = os.path.join(self.expt_res_dir, 'ckpt')
		self.create_dir(self.log_dir)
		self.create_dir(self.ckpt_dir)

	def set_config(self, config):
		for key in config:
			setattr(self, key, config[key])

	def __str__(self):
		res = ""
		for k in self.__dict__:
			res += "{}: {}\n".format(k, self.__dict__[k])
		return res

	def __repr__(self):
		res = ""
		for k in self.__dict__:
			res += "{}: {}\n".format(k, self.__dict__[k])
		return res

	def get_dict(self):
		return self.__dict__

	def create_dir(self, dir_path):

		if not os.path.exists(dir_path):
			os.makedirs(dir_path)	

# Creating an Object for the argument parser to populate the fields
args = Config()

def parse_args():

	parser = argparse.ArgumentParser(fromfile_prefix_chars = "@")

	# Experiment Related Options
	parser.add_argument('--log', action="store_true", default=False, help="Whether to log the results or not")
	parser.add_argument('--expt_res_base_dir', type=str, help="Path to base directory where all the results and logs related to the experiment will be stored")
	parser.add_argument('--expt_name', type=str, help="Name of the experiment to uniquely identify its folder")
	parser.add_argument('--mode', type=str, required=True, help="Specify the mode: {train, eval}")
	parser.add_argument('--learning_rate_decay_every', type=int, default=1000, help="The schedule after which the learning is decayed by half")
	parser.add_argument('--display_every', type=int, default=10, help="Loss statistics to display after every n batches")
	parser.add_argument('--drop_prob', default=0.0, type=float, help="Dropout probability for all linear layers")
	
	parser.add_argument('--nl', default='relu', choices=['relu', 'gated_tanh', 'tanh'], help="Type of Non linearity to be used in the network (relu, gated_tanh, tanh)")

	
	
	parser.add_argument('--weights_init', default='xavier', help="The initializer for weight matrices in the network")
	parser.add_argument('--gcn_depth', default=5, type=int, help="The depth of the GCN network")
	parser.add_argument('-ia','--initial_adders', default=2, type=int, help="Initial Adders")
	parser.add_argument('-gb','--gbottlenecks', default=2, type=int, help="The nums GBottleNecks in Deformer Block")
	

	# General system running and configuration options
	parser.add_argument('-l','--load_model_dirpath', type=str, default='', help='load model from path')
	parser.add_argument('-s','--save_model_dirpath', type=str, default='ckpt/', help='save model to path')
	parser.add_argument('-bs','--batch_size', type=int, default=1, help='Batch size ')
	parser.add_argument('--show_stat', type=int, default=1, help='Show stat at every batch')
	parser.add_argument('-sf','--sf', type=str, default='', help='suffix_name for pred')
	parser.add_argument('-n','--num_epochs', type=int, default=2000, help='The number of epochs for training the model')
	parser.add_argument('-d','--depth', type=int, default=0, help='Deformer depth')
	parser.add_argument('-lr','--lr', type=float, default=1e-5, help='See variable name')
	parser.add_argument('--step_size', type=int, default=100, help='See variable name')
	parser.add_argument('--gamma', type=float, default=0.8, help='See variable name')
	parser.add_argument('--lambda_n', type=float, default=1e-5, help='See variable name')
	parser.add_argument('--lambda_lap', type=float, default=0.2, help='See variable name')
	parser.add_argument('--lambda_e', type=float, default=0.1, help='See variable name')
	parser.add_argument('--train_dir', type=str, default='', help='See variable name')
	parser.add_argument('--test_dir', type=str, default='', help='See variable name')
	parser.add_argument('--val_dir', type=str, default='', help='See variable name')
	parser.add_argument('--suffix', type=str, default='train', help='See variable name')
	parser.add_argument('--feature_scale', type=int, default=10, help='See variable name')
	parser.add_argument('--dim_size', type=int, default=2, help='See variable name')
	parser.add_argument('--img_width', type=int, default=600, help='See variable name')
	parser.add_argument('--img_height', type=int, default=600, help='See variable name')
	parser.add_argument('-t','--test', dest='test', default = False, action='store_true',help='See variable name')
	parser.add_argument('--add_prob', type=float, default=0.5, help='See variable name')
	parser.add_argument('--num_polygons', type=int, default=3, help='See variable name')
	parser.add_argument('-i','--iters_per_block', type=int, default=100, help='See variable name')
	parser.add_argument('--load_rl_count', type=int, default=200, help='See variable name')



	return parser.parse_args(namespace = args)