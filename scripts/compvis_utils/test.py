#!/usr/bin/env python3
import numpy as np
import os
import six.moves.urllib as urllib
import sys

import tarfile

from collections import defaultdict
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from IPython.display import display
import ast
import cv2 as cv
# matplotlib.use('TkAgg')
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def d2r(alpha):
	theta = (pi/180)*alpha
	return theta

def r2d(theta):
	alpha = (180/pi)*theta
	return alpha

def get_ref_test_image_path():
	PROJECT_HOME_FOLDER = 	"../../"
	PATH_TO_REF_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/reference"
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	# print(REF_IMAGE_PATHS)	
	PATH_TO_TEST_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/test"
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

	# print(TEST_IMAGE_PATHS)
	#image = Image.open(TEST_IMAGE_PATHS[0])
	return PATH_TO_REF_IMAGES_DIR,PATH_TO_TEST_IMAGES_DIR


def main():
	DATABASE_PATH = sys.argv[1]
	# DATABASE_PATH = "/home/smarn/thesis/results/space_resection_testing/sr_tg5.txt"
	f = open(DATABASE_PATH,"rt")
	lines = f.readlines()
	data = []
	print(len(lines))
	i = 0
	omegais,phiis,omegas,phis,kappas,x0s,y0s,z0s = [],[],[],[],[],[],[],[]
	for i in range(0,len(lines),3):
		line1 = lines[i]
		line2 = lines[i+1]
		line3 = lines[i+2]
		# print(line2)
		init_omega = line1.split("|")[0].split(":")[1]
		init_phi = line1.split("|")[1].split(":")[1]
		x0 = line2.split("|")[0].split(":")[1]
		y0 = line2.split("|")[1].split(":")[1]
		z0 = line2.split("|")[2].split(":")[1]
		omega = line3.split("|")[0].split(":")[1]
		phi = line3.split("|")[1].split(":")[1]
		kappa = line3.split("|")[2].split(":")[1]
		if float(x0) >0 and float(y0)>0 and float(z0)>0:
			omegais.append(float(init_omega))
			phiis.append(float(init_phi))
			# print(omega,phi,kappa)
			# print(x0,y0,z0)
			x0s.append(round(float(x0),2))
			y0s.append(round(float(y0),2))
			z0s.append(round(float(z0),2))
			omegas.append(round(float(omega),2))
			phis.append(round(float(phi),2))
			kappas.append(round(float(kappa),2))
	# fig,ax = plt.subplots()
	# plt.scatter(omegas,x0s)
	# plt.show()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(np.array(omegais),np.array(phiis), np.array(x0s))
	ax.scatter(np.array(omegais),np.array(phiis), np.array(y0s))
	# ax.scatter(np.array(omegas),np.array(phis), np.array(z0s))
	ax.set_xlabel('INIT_OMEGAS')
	ax.set_ylabel('INIT_PHIS')
	plt.show()

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(np.array(omegais),np.array(phiis), np.array(omegas))
	ax.scatter(np.array(omegais),np.array(phiis), np.array(phis))
	ax.scatter(np.array(omegais),np.array(phiis), np.array(kappas))
	ax.set_xlabel('INIT_OMEGAS')
	ax.set_ylabel('INIT_PHIS')
	plt.show()

def chop_block_CV(or_image,block_bounds):
	# Usage: Crops an image to seperate image blocks (OPENCV)
	# Inputs: 1) Original image in np.array format (length,width,channels) BGR
	# 				2) Image block boundaries in image size percentages
	#	
	# Outputs:1) Cropped image np.array format (length,width,channels) BGR
	# 				2) Upper-left pixel coords of cropped image in original
	im_height,im_width,channels = or_image.shape
	ymin = block_bounds[0]
	xmin = block_bounds[1]
	ymax = block_bounds[2]	
	xmax = block_bounds[3]	
	left, bottom, right, top = int(xmin * im_width), int(ymax * im_height), int(xmax * im_width), int(ymin * im_height)
	cropped_im = or_image[top:bottom,left:right,:]
	# plt.imshow(cropped_im[:,:,::-1]),plt.show()
	origin = (left,top)
	return cropped_im,origin

def examine_sift_error():
	PATH_TO_REF_IMAGES_DIR,PATH_TO_TEST_IMAGES_DIR = get_ref_test_image_path()
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
	bounds = [0.39595363, 0.50751287, 0.6300319 , 0.67856157]
	image = cv.imread(TEST_IMAGE_PATHS[0],cv.COLOR_BGR2GRAY)
	cropped_im,origin = chop_block_CV(image,bounds)
	# plt.imshow(cropped_im[:,:,::-1]),plt.show()
	return cropped_im,origin

if __name__ == '__main__':
	# examine_sift_error()
	main()