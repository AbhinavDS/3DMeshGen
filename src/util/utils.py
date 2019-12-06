import numpy as np
import os
import sys
from PIL import Image, ImageDraw
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')
from src import MEAN, VAR, PAD_TOKEN

def project_1d(polygons_data_line,params):
	proj_data_line = np.zeros(params.img_width,dtype=float)
	feature_size = len(polygons_data_line[0])
	p = 0
	minx = params.img_width -1 
	maxx = 0
	while True:
		if p < feature_size-2 and polygons_data_line[0,p] == PAD_TOKEN and polygons_data_line[0,p+2] == PAD_TOKEN:
			proj_data_line[minx:maxx+1] = 1.0
			break
		if p >= feature_size:
			proj_data_line[minx:maxx+1] = 1.0
			break
		if polygons_data_line[0,p] == PAD_TOKEN:
			p += 2
			proj_data_line[minx:maxx+1] = 1.0
			minx = params.img_width -1
			maxx = 0
			continue
		minx = min(minx,int(polygons_data_line[0,p]))
		maxx = max(maxx,int(polygons_data_line[0,p]))
		p += 2
	proj_data_line = np.expand_dims(proj_data_line,axis = 0)
	return proj_data_line
