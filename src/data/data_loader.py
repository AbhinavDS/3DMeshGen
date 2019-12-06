import torch
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from src import MEAN, VAR, IMG_PAD_TOKEN, dtypeF
from src.util import utils

def seedRandom(seed):
	np.random.seed(seed)
	random.seed(seed)

def getMetaData(params, data_dir):
	f_meta = os.path.join(data_dir,'meta_%s.dat'%params.suffix)
	max_vertices = 0
	with open(f_meta) as f:
		line = f.readline().strip()
		max_vertices = int(line)
		line = f.readline().strip()
		data_size = int(line)
		line = f.readline().strip()
		max_total_vertices = int(line)
	return max_vertices, max_total_vertices*params.feature_scale*params.dim_size, data_size, max_total_vertices

def upsample(vertices, normals, edges, max_total_vertices):
	new_vertices = np.copy(vertices)
	new_normals = np.copy(normals)
	new_edges = edges.copy()

	adj_list = {}
	for i in range(max_total_vertices):
		adj_list[i] = set()

	for a,b in new_edges:
		adj_list[a].add(b)
		adj_list[b].add(a)

	for i in range(max_total_vertices - len(vertices)):
		new_vertex_id = len(new_vertices)
		a = random.randint(0, len(new_vertices)-1)
		b = random.sample(adj_list[a], 1)[0]
		adj_list[a].remove(b)
		adj_list[b].remove(a)		
		adj_list[a].add(new_vertex_id)
		adj_list[b].add(new_vertex_id)		
		adj_list[new_vertex_id].add(a)
		adj_list[new_vertex_id].add(b)		
		
		new_edges.remove((a,b))
		new_edges.remove((b,a))
		new_edges.add((a, new_vertex_id))
		new_edges.add((new_vertex_id, a))
		new_edges.add((b, new_vertex_id))
		new_edges.add((new_vertex_id, b))

		new_vertex = 0.5 * (new_vertices[a] + new_vertices[b])
		new_vertices = np.concatenate((new_vertices, np.expand_dims(new_vertex, axis=0)))
		
		new_normal = (new_normals[a] + new_normals[b])
		new_normal /= np.linalg.norm(new_normal)
		new_normals = np.concatenate((new_normals, np.expand_dims(new_normal, axis=0)))
	return new_vertices, new_normals, new_edges

def getPointsAndEdges(params, polygons_data_line, normals_data_line):
	reshaped_polygon = polygons_data_line.reshape((-1, params.dim_size))
	reshaped_normals = normals_data_line.reshape((-1, params.dim_size))
	edges = set()
	i = 0
	start_vertex = 0
	num_polygon = 0
	while(i < reshaped_polygon.shape[0]):
		if i == start_vertex:
			pass
		elif reshaped_polygon[i][0] != -2:
			a = i-1-num_polygon
			b = i-num_polygon
			# print(f'{a} -> {b}')
			edges.add((a,b))
			edges.add((b,a))
		elif reshaped_polygon[i][0] == -2:
			a = i-1-num_polygon
			b = start_vertex-num_polygon
			# print(f'{a} -> {b}')
			edges.add((a,b))
			edges.add((b,a))
			start_vertex = i + 1
			num_polygon += 1
		if i == reshaped_polygon.shape[0] - 1:
			a = i-num_polygon
			b = start_vertex-num_polygon
			# print(f'{a} -> {b}')
			edges.add((a,b))
			edges.add((b,a))
			num_polygon += 1
		i += 1
	
	valid_vertices = reshaped_polygon[:,0] != -2
	reshaped_polygon = reshaped_polygon[valid_vertices]
	reshaped_normals = reshaped_normals[valid_vertices]
	return reshaped_polygon, reshaped_normals, edges 

def getDataLoader(params, data_dir, max_total_vertices, feature_size):
	f_polygons_path = os.path.join(data_dir,'polygons_%s.dat'%params.suffix)
	f_normals_path = os.path.join(data_dir,'normals_%s.dat'%params.suffix)
	iter_count = 0
	polygons_data = np.array([])
	normals_data = np.array([])
	proj_data = np.array([])
	seq_len = np.array([])
	while True:

		with open(f_polygons_path, 'r') as f_polygons:
			with open(f_normals_path, 'r') as f_normals:
				polygons_line = f_polygons.readline()
				normals_line = f_normals.readline()

				while(polygons_line != '' or normals_line != ''):
					iter_count += 1

					# Get polygons and normals
					polygons_data_line = np.fromstring(polygons_line, dtype=float, sep=',')
					normals_data_line = np.fromstring(normals_line, dtype=float, sep=',')

					image_features = np.expand_dims(np.pad(polygons_data_line,(0,feature_size-len(polygons_data_line)),'constant',constant_values=(0,IMG_PAD_TOKEN)),0)
					
					# Get Polygons, Normals, and Edge List; Also upsample points. Also scale points within -1, 1
					polygons, normals, edges = getPointsAndEdges(params, polygons_data_line, normals_data_line)
					polygons, normals, edges = upsample(polygons, normals, edges, max_total_vertices)
					polygons = (polygons - MEAN)/VAR
					polygons = np.expand_dims(polygons, 0)
					normals = np.expand_dims(normals, 0)
					assert(params.dim_size == 2)
					
					#1d projection of 2d mesh
					proj_data_line = utils.project_1d(np.expand_dims(polygons_data_line, 0),params)			
			
					if(len(polygons_data)==0):
						polygons_data = polygons
						normals_data = normals
						edges_data = np.array([edges])
						image_data = image_features
						proj_data = proj_data_line
					else:
						polygons_data = np.concatenate((polygons_data, polygons),axis=0)
						normals_data = np.concatenate((normals_data, normals),axis=0)
						edges_data = np.concatenate((edges_data, np.array([edges])),axis=0)
						image_data = np.concatenate((image_data, image_features),axis=0)
						proj_data = np.concatenate((proj_data,proj_data_line),axis=0)

					if iter_count >= params.batch_size:
						yield polygons_data, normals_data, edges_data, image_data, proj_data
						iter_count = 0
						polygons_data = np.array([])
						normals_data = np.array([])
						edges_data = np.array([])
						image_data = np.array([])
						proj_data = np.array([])

					polygons_line = f_polygons.readline()
					normals_line = f_normals.readline()