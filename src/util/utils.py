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


def scaleBackTensor(c):
	return (c*VAR + MEAN)

def scaleBack(c):
	return (c*VAR + MEAN).tolist()

def drawPolygons(polygons, polygonsgt, edgesgt, proj_pred=None, proj_gt=None, color='red',out='out.png',A=None, line=None, text=None):
	black = (0,0,0)
	white=(255,255,255)
	im = Image.new('RGB', (600, 620), white)
	imPxAccess = im.load()
	draw = ImageDraw.Draw(im,'RGBA')
	verts = polygons
	vertsgt = polygonsgt

	# Pred
	# either use .polygon(), if you want to fill the area with a solid colour
	points = tuple(tuple(x) for x in verts)
	#draw.point((points),fill=(255,0,0,0))
	for point in points:
	    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill=color)
	if A is None:
		draw.polygon((points), outline=black,fill=(0,0,0,0) )
	else:
	# # or .line() if you want to control the line thickness, or use both methods together!
		for i in range(len(verts)):
			for j in range(len(verts)):
				if(A[i,j]):					
					draw.line((tuple(verts[i]),tuple(verts[j])), width=2, fill=black )
	
	# GT 
	for a, b in edgesgt:
		draw.ellipse((polygonsgt[a][0] - 4, polygonsgt[a][1] - 4, polygonsgt[a][0]  + 4, polygonsgt[a][1] + 4), fill='green')
		draw.ellipse((polygonsgt[b][0] - 4, polygonsgt[b][1] - 4, polygonsgt[b][0]  + 4, polygonsgt[b][1] + 4), fill='green')
		draw.line((tuple(polygonsgt[a]),tuple(polygonsgt[b])), width=2, fill='green' )

	# Shadow
	if proj_gt is not None:
		for i in range(im.size[0]):
			for j in range(im.size[1]-10,im.size[1]):
				imPxAccess[i,j] = (0,int(proj_gt[i])*128,0)
	if proj_pred is not None:
		for i in range(im.size[0]):
			for j in range(im.size[1]-20,im.size[1]-10):
				if color == 'red':
					imPxAccess[i,j] = (int(proj_pred[i])*255,0,0)
				elif color == 'blue':
					imPxAccess[i,j] = (0,0,int(proj_pred[i])*255)
				else:
					imPxAccess[i,j] = (int(proj_pred[i])*255,int(proj_pred[i])*255,0)

	if line is not None:
		x1,y1,x2,y2 = line
		x1 = x1*VAR + MEAN
		y1 = y1*VAR + MEAN
		x2 = x2*VAR + MEAN
		y2 = y2*VAR + MEAN
		draw.line(((x1,y1),(x2,y2)), width=5, fill=black)
		if text is not None:
			draw.text((10,10), f'({x1:2f},{y1:2f}) -> ({x2:2f},{y2:2f}) {text}', fill=black)
		else:
			draw.text((10,10), f'({x1:2f},{y1:2f}) -> ({x2:2f},{y2:2f})', fill=black)
	im.save(out)


def flatten_pred(c,A,params):
	#c:  num_vertsx2
	#A:  num_vertsxnum_verts
	num_verts = c.shape[0]
	vertFlags = np.zeros(num_verts)
	proj_data_line = np.zeros(params.img_width,dtype=float)
	v = 0
	minx = params.img_width - 1
	maxx = 0
	
	start = v
	minx = min(minx,c[v,0])
	maxx = max(maxx,c[v,0])
	vertFlags[v] = 1
	for j in range(num_verts):
		if A[v,j] and vertFlags[j]==0:
			v = j
			break
	while True:
		minx = min(minx,c[v,0])
		maxx = max(maxx,c[v,0])
		vertFlags[v] = 1
		found_nbr = False
		for j in range(num_verts):
			if A[v,j] and vertFlags[j]==0:
				v = j
				found_nbr = True
				break
		if not found_nbr:
			proj_data_line[minx:maxx+1] = 1.0
			minx = params.img_width - 1
			maxx = 0
			found_poly = False
			for j in range(num_verts):
				if vertFlags[j]==0:
					v = j
					found_poly = True
					break
			if not found_poly:
				break
	return proj_data_line
