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
import time 

sys.path.append('..')
matplotlib.use('TkAgg')
global case


def evaluate_all():
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	RESULTS_MORN = "../../evaluation/results_morning.txt"
	RESULTS_MID = "../../evaluation/results_midday.txt"
	RESULTS_AFT = "../../evaluation/results_afternoon.txt"
	f_mor = open(RESULTS_MORN,"rt")
	f_mid = open(RESULTS_MID,"rt")
	f_aft = open(RESULTS_AFT,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_morn = f_mor.readlines()
	res_mid = f_mid.readlines()
	res_aft = f_aft.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,image,norms_morn = [],[],[],[]
	for i in range(1,len(ground_lines)):
		image.append(ground_lines[i].split('|')[0])
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_morn[i].split('|')[2].split(',')[0])
		res_y = float(res_morn[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_morn.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	x_diff,y_diff,image,norms_mid = [],[],[],[]
	for i in range(1,len(ground_lines)):
		image.append(ground_lines[i].split('|')[0])
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_mid[i].split('|')[2].split(',')[0])
		res_y = float(res_mid[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_mid.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	x_diff,y_diff,image,norms_aft = [],[],[],[]
	for i in range(1,len(ground_lines)):
		image.append(ground_lines[i].split('|')[0])
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_aft[i].split('|')[2].split(',')[0])
		res_y = float(res_aft[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_aft.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	fig,ax = plt.subplots(1,1)
	plt.title("Σφαλματα Μεθόδου για όλες τις θέσεις και ώρες Λήψης.")
	ax.set_ylim(0.0,1.4)
	plt.plot(image,norms_morn)
	plt.scatter(image,norms_morn)
	plt.plot(image,norms_mid)
	plt.scatter(image,norms_mid)
	plt.plot(image,norms_aft)
	plt.scatter(image,norms_aft)
	ax.grid("on")
	ax.set_ylabel("Σφάλμα (m)")
	ax.set_xlabel("ID εικόνας")
	ax.legend(["Πρωινή","Μεσημεριανή","Απογευματινή"])
	plt.show()

def evaluate(RESULTS_PATH):
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	f_results = open(RESULTS_PATH,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_lines = f_results.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,image,norms = [],[],[],[]
	sum_res = 0
	for i in range(1,len(ground_lines)):
		image.append(ground_lines[i].split('|')[0])
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms.append(norm)
		sum_res += norm
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	print("mean err {}".format(sum_res/33))
	fig,ax = plt.subplots(1,1)
	plt.title("{} ΛΗΨΗ.\nΣφαλματα Μεθόδου για όλες τις θέσεις Λήψης.".format(case))
	ax.set_ylim(0.0,1.4)
	plt.plot(image,norms)
	plt.scatter(image,norms)
	ax.grid("on")
	ax.set_ylabel("Σφάλμα (m)")
	ax.set_xlabel("ID εικόνας")
	# ax.legend(["DOOR1","DOOR2-DOOR3","DOOR5"])
	plt.show()


def evaluate_rand(RESULTS_PATH):
	# RESULTS_PATH = "../../evaluation/results_afternoon.txt"
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	f_results = open(RESULTS_PATH,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_lines = f_results.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,image,norms_1 = [],[],[],[]
	for i in range(28,34):
		image.append(ground_lines[i].split('|')[0])
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
	fig,ax = plt.subplots(1,1)
	plt.title("{} ΛΗΨΗ.\nΈλεγχος μεθόδου για τυχαίες θέσεις.".format(case))
	ax.set_ylim(0.0,1.4)
	ax.plot(image,norms_1)
	# ax.plot(dist_2,norms_2)
	ax.scatter(image,norms_1)
	# ax.scatter(dist_2,norms_2)
	ax.grid("on")
	ax.set_ylabel("Σφάλμα (m)")
	ax.set_xlabel("Image ID")
	# ax.legend(["DOOR4","DOOR9"])
	plt.show()


def evaluate_angle(RESULTS_PATH):
	# RESULTS_PATH = "../../evaluation/results_afternoon.txt"
	GROUND_TRUTH_PATH = "../../database/ground_truths.txt"
	f_results = open(RESULTS_PATH,"rt")
	f_ground = open(GROUND_TRUTH_PATH,"rt")
	res_lines = f_results.readlines()
	ground_lines = f_ground.readlines()
	x_diff,y_diff,angle_1,norms_1 = [],[],[],[]
	for i in range(15,20):
		angle_1.append(ground_lines[i].split('|')[-2])
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
		angle_2.append(ground_lines[i].split('|')[-2])
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
		angle_3.append(ground_lines[i].split('|')[-2])
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_3.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))
	# plt.scatter(angle,x_diff)
	fig,ax = plt.subplots(1,1)
	plt.title("{} ΛΗΨΗ.\nΈλεγχος μεθόδου από διαφερικές γωνίες λήψης.".format(case))
	ax.set_ylim(0.0,1.4)
	plt.plot(angle_1,norms_1)
	plt.plot(angle_2,norms_2)
	plt.plot(angle_3,norms_3)
	plt.scatter(angle_1,norms_1)
	plt.scatter(angle_2,norms_2)
	plt.scatter(angle_3,norms_3)
	ax.grid("on")
	ax.set_ylabel("Σφάλμα (m)")
	ax.set_xlabel("Γωνία Λήψης - Θ (°)")
	ax.legend(["DOOR1","DOOR2","DOOR5"])
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
		dist_1.append(ground_lines[i].split('|')[-3])
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
		dist_2.append(ground_lines[i].split('|')[-3])
		g_x = float(ground_lines[i].split('|')[2].split(',')[0])
		g_y = float(ground_lines[i].split('|')[2].split(',')[1])
		res_x = float(res_lines[i].split('|')[2].split(',')[0])
		res_y = float(res_lines[i].split('|')[2].split(',')[1])
		norm = sqrt(pow(g_x-res_x,2)+pow(g_y-res_y,2))
		norms_2.append(norm)
		x_diff.append(abs(g_x-res_x))
		y_diff.append(abs(g_y-res_y))

	fig,ax = plt.subplots(1,1)
	plt.title("{} ΛΗΨΗ.\nΈλεγχος μεθόδου από διαφερικές αποστάσεις λήψης.".format(case))
	ax.set_ylim(0.0,1.4)
	ax.plot(dist_1,norms_1)
	ax.plot(dist_2,norms_2)
	ax.scatter(dist_1,norms_1)
	ax.scatter(dist_2,norms_2)
	ax.grid("on")
	ax.set_ylabel("Σφάλμα (m)")
	ax.set_xlabel("Απόσταση (m)")
	ax.legend(["DOOR4","DOOR9"])
	plt.show()

def evaluate_runtimes(RESULTS_PATH):
	f_results = open(RESULTS_PATH,"rt")
	res_lines = f_results.readlines()
	t_nn,t_cv,t_sp,t_all = [],[],[],[]
	image = []
	sum_t = 0
	for i in range(1,len(res_lines)):
		print(res_lines[i].split('|')[-2])
		image.append(int(res_lines[i].split('|')[0]))
		tnn = float(res_lines[i].split('|')[-2].split(',')[0])
		tcv = float(res_lines[i].split('|')[-2].split(',')[1])
		tsp = float(res_lines[i].split('|')[-2].split(',')[2])
		tall = float(res_lines[i].split('|')[-2].split(',')[3])
		t_nn.append(tnn)
		t_cv.append(tcv)
		t_sp.append(tsp)
		sum_t += tall
		t_all.append(tall)
	# plt.scatter(image,x_diff)
	fig,ax = plt.subplots(2,2)
	fig.suptitle("{} ΛΗΨΗ,\nΧρόνοι Εκτέλεσης.".format(case),fontweight="bold")
	ax[0,0].plot(image,t_nn)
	ax[0,1].plot(image,t_cv)
	ax[1,0].plot(image,t_sp)
	ax[1,1].plot(image,t_all)
	ax[0,0].scatter(image,t_nn)
	ax[0,1].scatter(image,t_cv)
	ax[1,0].scatter(image,t_sp)
	ax[1,1].scatter(image,t_all)
	ax[0,0].grid("on")
	ax[0,1].grid("on")
	ax[1,0].grid("on")
	ax[1,1].grid("on")
	ax[0,0].set_ylim(0,50)
	ax[0,1].set_ylim(0,50)
	# ax[1,0].set_ylim(0,50)
	ax[1,1].set_ylim(0,50)
	ax[0,0].set_title("1.Νευρωνικό",fontweight="bold")
	ax[0,1].set_title("2.Αλγόριθμος Εξαγωγής \nΣημείων Ελέγχου",fontweight="bold")
	ax[1,0].set_title("3.Φωτογραμμετρική Οπισθοτομία",fontweight="bold")
	ax[1,1].set_title("4.Συνολικός Χρόνος",fontweight="bold")
	ax[0,0].set_ylabel("Χρόνος (s)")
	ax[1,0].set_ylabel("Χρόνος (s)")
	ax[1,0].set_xlabel("ID εικόνας")
	ax[1,1].set_xlabel("ID εικόνας")
	print(sum_t)
	# ax.legend(["DOOR1","DOOR2-DOOR3","DOOR5"])
	# plt.show()

	# fig,ax = plt.subplots(1,1)
	# fig.suptitle("{} ΛΗΨΗ,\nΧρόνοι Εκτέλεσης.".format(case))
	# ax[0,0].plot(image,t_nn)
	# ax[0,1].plot(image,t_cv)
	# ax[1,0].plot(image,t_sp)
	# ax[1,1].plot(image,t_all)
	# ax[0,0].scatter(image,t_nn)
	# ax[0,1].scatter(image,t_cv)
	# ax[1,0].scatter(image,t_sp)
	# ax[1,1].scatter(image,t_all)
	# ax[0,0].grid("on")
	# ax[0,1].grid("on")
	# ax[1,0].grid("on")
	# ax[1,1].grid("on")
	# ax[0,0].set_ylim(0,50)
	# ax[0,1].set_ylim(0,50)
	# # ax[1,0].set_ylim(0,50)
	# ax[1,1].set_ylim(0,50)
	# ax[0,0].set_title("Νευρωνικό",fontweight="bold")
	# ax[0,1].set_title("OpenCV")
	# ax[1,0].set_title("Space Resection")
	# ax[1,1].set_title("Over All")
	# ax[0,0].set_ylabel("Χρόνος (s)")
	# ax[1,0].set_ylabel("Χρόνος (s)")
	# ax[1,0].set_xlabel("ID εικόνας")
	# ax[1,1].set_xlabel("ID εικόνας")
	# ax.legend(["DOOR1","DOOR2-DOOR3","DOOR5"])
	plt.show()



if __name__ == '__main__':
	global case
	RESULTS_PATH = "../../evaluation/results_midday.txt"
	if "results_morning.txt" in RESULTS_PATH.split("/"):
		case = "ΠΡΩΙΝΗ"
	elif "results_midday.txt" in RESULTS_PATH.split("/"):
		case = "ΜΕΣΗΜΕΡΙΑΝΗ"
	else:
		case = "ΑΠΟΓΕΥΜΑΤΙΝΗ"
	# evaluate(RESULTS_PATH)
	# evaluate_rand(RESULTS_PATH)
	# evaluate_angle(RESULTS_PATH)
	# evaluate_dist(RESULTS_PATH)
	evaluate_runtimes(RESULTS_PATH)
	# evaluate_all()