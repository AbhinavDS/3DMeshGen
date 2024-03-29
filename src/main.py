"""
Main module that controls the complete graph QA pipeline
"""

import math
from config_parser import parse_args
from trainer import Trainer
from evaluator import Evaluator
from data.data_loader import getMetaData, getDataLoader

if __name__ == "__main__":

	args = parse_args()
	args.gen_config()

	# Calculate Meta Parameters from dataset
	train_max_vertices, train_feature_size, train_data_size, train_max_total_vertices = getMetaData(args, args.train_dir)
	val_max_vertices, val_feature_size, val_data_size, val_max_total_vertices = getMetaData(args, args.val_dir)
	test_max_vertices, test_feature_size, test_data_size, test_max_total_vertices = getMetaData(args, args.test_dir)
	args.max_vertices = max(train_max_vertices, val_max_vertices, test_max_vertices)
	args.feature_size = max(train_feature_size, val_feature_size, test_feature_size)
	args.data_size = max(train_data_size, val_data_size, test_data_size)
	args.max_total_vertices = max(train_max_total_vertices, val_max_total_vertices, test_max_total_vertices)
	args.train_data_size = train_data_size
	args.val_data_size = val_data_size
	args.test_data_size = test_data_size

	# Adjust args.gbottlenecks accordingly
	args.gbottlenecks = int(math.log(args.max_total_vertices))


	if args.mode == "train":	
		
		
		train_generator = getDataLoader(args, args.train_dir, args.max_total_vertices, args.feature_size)
		val_generator = getDataLoader(args, args.val_dir, args.max_total_vertices, args.feature_size)
		print(args)
		trainer = Trainer(args, train_generator, val_generator)
		trainer.train()
	
	elif args.mode == "eval":
		
		test_generator = getDataLoader(args, args.test_dir, args.max_total_vertices, args.feature_size)
		print(args)
		evaluator = Evaluator(args, test_generator)
		evaluator.eval()
		pass
	else:
		raise("Please specify the correct training/testing mode")