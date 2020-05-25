#!/usr/bin/env python3
import pathlib

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
from numpy.linalg import inv

from math import *
import yaml


def draw_circle(img,centre,COLOUR):
    # img = cv.imread(img_path)
    THICKNESS = 10
    img = cv.circle(img,centre,50,COLOUR,THICKNESS)
    img = cv.circle(img,centre,3,COLOUR,THICKNESS)

def load_intrincis(case="pixel"):
	pixel_size = 1.12e-6
	with open(r'../../camera_caliblation/camera_intrinsics.yaml') as file:
		documents = yaml.full_load(file)
	for item, doc in documents.items():
		if item == "coordinates":
			coord_sys = doc
		if item == "focalx":
			fx = int(doc)
		if item == "focaly":
			fy = int(doc)
		if item == "ppoffsetx":
			x0 = int(doc)
		if item == "ppoffsety":
			y0 = int(doc)
	f = int((fx + fy)/2)
	if case != coord_sys:
		x0 = round(x0*pixel_size,5)
		y0 = round(y0*pixel_size,5)
		f = round(f*pixel_size,5)
	return x0,y0,f

def space_resection(space_coords,picture_coords,init_eop,iop):
	res_cur = 1000000000
	res_old = 1000000000
	xa = picture_coords[:,0]
	ya = picture_coords[:,1]
	
	XA = space_coords[:,0]
	YA = space_coords[:,1]
	ZA = space_coords[:,2]

	dof = 2*len(XA) - 6

	omega = init_eop[0]
	phi = init_eop[1]
	kappa = init_eop[2]
	X0 = init_eop[3]
	Y0 = init_eop[4]
	Z0 = init_eop[5]
	
	x0 = iop[0]
	y0 = iop[1]
	f = iop[2]

	delta = np.array([1,1,1,1,1,1])

	num_points = len(XA)
	ii = 0
	while max(abs(delta)) > 0.00001:
		# print(delta)
		ii += 1
		print(ii)
		mw = np.matrix([[1,0,0],[0,cos(omega),sin(omega)],[0,-sin(omega),cos(omega)]])
		mp = np.matrix([[cos(phi),0,-sin(phi)],[0,1,0],[sin(phi),0,cos(phi)]])
		mk = np.matrix([[cos(kappa),sin(kappa),0],[-sin(kappa),cos(kappa),0],[0,0,1]])

		m = mk*mp*mw

		dx = np.zeros(num_points)
		dy = np.zeros(num_points)
		dz = np.zeros(num_points)
		q = np.zeros(num_points)
		r = np.zeros(num_points)
		s = np.zeros(num_points)
		for i in range(num_points):
			dx[i] = XA[i] - X0
			dy[i] = YA[i] - Y0
			dz[i] = ZA[i] - Z0
			q[i] = m[2,0]*(XA[i] - X0) + m[2,1]*(YA[i] - Y0) + m[2,2]*(ZA[i] - Z0)
			r[i] = m[0,0]*(XA[i] - X0) + m[0,1]*(YA[i] - Y0) + m[0,2]*(ZA[i] - Z0)
			s[i] = m[1,0]*(XA[i] - X0) + m[1,1]*(YA[i] - Y0) + m[1,2]*(ZA[i] - Z0)

		double_length = num_points * 2
		j = 0
		L = np.mat(np.zeros((double_length,1)))
		for k in range(0,double_length,2):
			# L[k,0] =  xa[j]
			# L[k+1,0] =  ya[j]
			L[k,0] =  xa[j] - (x0 - f*r[j]/q[j])
			L[k+1,0] =  ya[j] - (y0 - f*s[j]/q[j])
			j += 1
		
		j = 0
		A = np.mat(np.zeros((double_length,6)))
		for k in range(0,double_length,2):
			A[k,0] = (f/pow(q[j],2)) * (r[j] * (-m[2,2]*dy[j] + m[2,1]*dz[j]) - q[j] * (-m[0,2]*dy[j] + m[0,1]*dz[j]))
			A[k,1] = (f/pow(q[j],2)) *(r[j]*(dx[j]*cos(phi) + dy[j]*sin(omega)*sin(phi) - dz[j]*sin(phi)*cos(omega)) - (q[j])*(-dx[j]*sin(phi)*cos(kappa) + dy[j]*sin(omega)*cos(phi)*cos(kappa) - dz[j]*cos(omega)*cos(phi)*cos(kappa)))
			A[k,2] = -(f/q[j]) * (m[1,0]*dx[j]+m[1,1]*dy[j]+m[1,2]*dz[j])
			A[k,3] = -(f/pow(q[j],2))*(r[j]*m[2,0] + q[j]*m[0,0])
			A[k,4] = -(f/pow(q[j],2))*(r[j]*m[2,1] + q[j]*m[0,1])
			A[k,5] = -(f/pow(q[j],2))*(r[j]*m[2,2] + q[j]*m[0,2])
			A[k+1,0] = (f/pow(q[j],2)) * (s[j] * (-m[2,2]*dy[j] + m[2,1]*dz[j]) - q[j] * (-m[1,2]*dy[j] + m[1,1]*dz[j]))
			A[k+1,1] = (f/pow(q[j],2)) *(s[j]*(dx[j]*cos(phi) + dy[j]*sin(omega)*sin(phi) - dz[j]*sin(phi)*cos(omega)) - (q[j])*(-dx[j]*sin(phi)*sin(kappa) - dy[j]*sin(omega)*cos(phi)*sin(kappa) + dz[j]*cos(omega)*cos(phi)*sin(kappa)))
			A[k+1,2] = (f/q[j]) * (m[0,0]*dx[j]+m[0,1]*dy[j]+m[0,2]*dz[j])
			A[k+1,3] = -(f/pow(q[j],2))*(s[j]*m[2,0] - q[j]*m[1,0])
			A[k+1,4] = -(f/pow(q[j],2))*(s[j]*m[2,1] - q[j]*m[1,1])
			A[k+1,5] = -(f/pow(q[j],2))*(s[j]*m[2,2] - q[j]*m[1,2])
			j += 1

		# delta = [0]*(9+4)
		delta = ((A.T*A).I)*A.T*L
		V = A*delta - L
		res_cur = sqrt((V.T*V)/dof)
		if res_cur > res_old:
			print(res_cur)
			res_old = res_cur
			# if ii > 2:
				# break
		else:
			res_old = res_cur
		# print(delta)
		# ml = np.transpose(b)*b
		# btb = inv(ml)
		# btf = np.transpose(b)*np.transpose(L)
		# delta = btb*btf
		print(delta)
		omega = omega + delta[0,0]
		phi = phi + delta[1,0]
		kappa = kappa + delta[2,0]
		X0 = X0 + delta[3,0]
		Y0 = Y0 + delta[4,0]
		print("Z0 prin:",Z0)
		Z0 = Z0 + delta[5,0]
		print("Z0 meta:",Z0)
		if ii == 100:
			break
	if ii > 100:
		return (omega,phi,kappa,X0,Y0,Z0)
	else:
		return (omega,phi,kappa,X0,Y0,Z0)

def collinearity_eqn(iop,eop,point):
    X = point[0]
    Y = point[1]
    Z = point[2]
    x0 = iop[0]
    y0 = iop[1]
    focallength = iop[2]

    om = eop[0]
    ph = eop[1]
    kp = eop[2]

    XL = eop[3]
    YL = eop[4]
    ZL = eop[5]

    Mom = np.matrix([[1, 0, 0], [0, cos(om), sin(om)], [0, -sin(om), cos(om)]])
    Mph = np.matrix([[cos(ph), 0, -sin(ph)], [0, 1, 0], [sin(ph), 0, cos(ph)]])
    Mkp = np.matrix([[cos(kp), sin(kp), 0], [-sin(kp), cos(kp), 0], [0, 0, 1]])

    M = Mkp * Mph * Mom
    # print(M)
    uvw = M * np.matrix([[X-XL], [Y-YL], [Z-ZL]])
    # print(uvw)
    x = x0 - (focallength * uvw[0,0] / uvw[2,0])
    y = y0 - (focallength * uvw[1,0] / uvw[2,0])

    return x, y

def pre_processing(image_coords_raw):
	x0 = 2304
	y0 = 1728
	Width_ratio = 5.12 / 4608
	Length_ratio = 3.84 / 3456
	image_coords = []
	for CPT in image_coords_raw:
		xp = CPT[0] - x0
		yp = CPT[1] - y0
		xp = xp * Width_ratio
		yp = yp * Length_ratio
		image_coords.append((xp,yp))
	image_coords = np.array(image_coords)
	return image_coords

def test1():
	image = [(1347, 966), (1766, 975), (2206, 985), (1354, 1466), (1766, 1476), (2199, 1486), (1372, 1978), (1766, 1999), (2203, 2020), (586, 802), (856, 882), (575, 2236), (843, 2120)]
	space = [(0, 130, 2170), (0, 1050, 2170), (0, 1970, 2170), (0, 130, 1085), (0, 1050, 1085), (0, 1970, 1085), (0, 130, 0), (0, 1050, 0), (0, 1970, 0), (2425, 0, 2140), (1550, 0, 2140), (2425, 0, 0), (1550, 0, 0)]
	space_coords = np.array(space)
	image_coords = pre_processing(image)
	f = 26
	X0 = 7000
	Y0 = 2000
	Z0 = 1000
	omega = (PI/180)*0
	phi = (PI/180)*90
	kappa = (PI/180)*0
	initial_coords = [omega,phi,kappa,X0,Z0,Y0]
	# print(image_coords)
	# print(space_coords)
	# print(initial_coords)
	# print(image_coords)
	# camera_coords = main(space_coords,image_coords,initial_coords,f)
	# print(camera_coords)
	# main1()
	iop = [2.56,1.92,26]
	eop = [0,1.57,0,10350,1670,1630]
	x,y = collinearity_eqn(iop,eop,0, 1050, 1085)
	print(x,y)	

def main2():
	x0,y0,f = load_intrincis()
	iop = [x0,y0,f]
	eop = [pi,0,pi/2,0,0,0.29]
	X = [[-0.05,-0.1,0],[-0.05,0,0],[-0.05,0.1,0],[0,-0.1,0],[0,0,0],[0,0.1,0],[0.05,-0.1,0],[0.05,0,0],[0.05,0.1,0]]
	img = cv.imread("../../images/HOME/vertical.jpg")
	i = 50
	for point in X:
		# point = point*1000
		x,y = collinearity_eqn(iop,eop,point)
		# iop = [0.013,-0.007,153.916]
		# rotation_matrix1(1.5,3.14,1)

		# eop = [dtr(-0.3952),dtr(1.2095),dtr(102.8006),1027.863,1043.998,611.032]
		# x,y = collinearity_eqn(iop,eop,974.435, 956.592, 14.619)
		# x = round(x,5)
		# y = round(y,5)
		draw_circle(img,(int(x),int(y-100)),(0,0,i))
		i += 25
		print(int(x),int(y-100))
	plt.imshow(img[:,:,::-1]),plt.show()


if __name__ == '__main__':
	# main2()
	XYZ = np.array([[-0.05,-0.1,0],[-0.05,0,0],[-0.05,0.1,0],[0,-0.1,0],[0,0,0],[0,0.1,0],[0.05,-0.1,0],[0.05,0,0],[0.05,0.1,0]])
	# xy = np.array([[975,991],[2225,991],[3474,991],[975,1616],[2225,1616],[3474,1616],[975,2240],[2225,2240],[3474,2240]])
	xy = np.array([[486, 1600],
								[1736, 1600],
								[2985, 1600],
								[486 ,2225],
								[1736, 2225],
								[2985, 2225],
								[486 ,2849],
								[1736 ,2849],
								[2985 ,2849]])
	pixel_size = 1.12e-6
	# xy = xy * pixel_size
	x0,y0,f = load_intrincis()
	iop = [x0,y0,f]
	init_eop = [3.0,0,1.51,0.1,0.1,0.5]
	res = space_resection(XYZ,xy,init_eop,iop)
	print(res)
	# img = cv.imread("../../images/HOME/vertical.jpg")
	# draw_circle(img,(1000,1000),(0,0,100))
	# plt.imshow(img[:,:,::-1]),plt.show()



