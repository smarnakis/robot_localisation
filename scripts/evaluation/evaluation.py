import sys
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2 as cv
import time 
import glob
from math import *

sys.path.append('..')
matplotlib.use('TkAgg')

def evaluate(RESULTS_PATH):
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	f_results = open(RESULTS_PATH,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_lines = f_results.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,image,norms = [],[],[],[]
	for i in range(1,len(ground_lines)):
		image.append(int(ground_lines[i].split('|')[0]))
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	# plt.scatter(image,x_diff)
	plt.plot(image,norms)
	plt.scatter(image,norms)
	plt.show()
	# plt.scatter(image,y_diff)
	# plt.scatter(image,norms)
	# plt.show()

def evaluate_angle(RESULTS_PATH):
	# RESULTS_PATH = "../../evaluation/results_afternoon.txt"
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	f_results = open(RESULTS_PATH,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_lines = f_results.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,angle_1,norms_1 = [],[],[],[]
	for i in range(15,20):
		angle_1.append(int(ground_lines[i].split('|')[-2]))
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_1.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	# plt.scatter(angle_1,norms_1)
	# plt.show()
	x_diff,y_diff,angle_2,norms_2 = [],[],[],[]
	for i in range(20,24):
		angle_2.append(int(ground_lines[i].split('|')[-2]))
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_2.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	
	# plt.show()	
	x_diff,y_diff,angle_3,norms_3 = [],[],[],[]
	for i in range(24,28):
		angle_3.append(int(ground_lines[i].split('|')[-2]))
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_3.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	# plt.scatter(angle,x_diff)
	fig = plt.figure(1)
	plt.plot(angle_1,norms_1)
	plt.plot(angle_2,norms_2)
	plt.plot(angle_3,norms_3)
	plt.scatter(angle_1,norms_1)
	plt.scatter(angle_2,norms_2)
	plt.scatter(angle_3,norms_3)
	plt.show()

def evaluate_dist(RESULTS_PATH):
	# RESULTS_PATH = "../../evaluation/results_afternoon.txt"
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	f_results = open(RESULTS_PATH,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_lines = f_results.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,dist_1,norms_1 = [],[],[],[]
	for i in range(1,8):
		dist_1.append(float(ground_lines[i].split('|')[-3]))
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_1.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	# plt.scatter(dist_1,norms_1)
	# plt.show()
	x_diff,y_diff,dist_2,norms_2 = [],[],[],[]
	for i in range(8,15):
		dist_2.append(float(ground_lines[i].split('|')[-3]))
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_2.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))

	fig = plt.figure(1)
	plt.plot(dist_1,norms_1)
	plt.plot(dist_2,norms_2)
	plt.scatter(dist_1,norms_1)
	plt.scatter(dist_2,norms_2)
	plt.show()

if __name__ == '__main__':
	RESULTS_PATH = "../../evaluation/results_midday.txt"
	evaluate(RESULTS_PATH)
	evaluate_angle(RESULTS_PATH)
	evaluate_dist(RESULTS_PATH)