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
# matplotlib.use('TkAgg')
import numpy as np
from math import *
import yaml
import time
pi = 3.1416

def d2r(alpha):
	theta = (pi/180)*alpha
	return theta

def r2d(theta):
	alpha = (180/pi)*theta
	return alpha

def load_intrincis():
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
	return x0,y0,f

def space_resection(XYZ,xy,eop,f):
	xp = xy[:,0]
	yp = xy[:,1]
	
	x = XYZ[:,0]
	y = XYZ[:,1]
	z = XYZ[:,2]

	ng = len(z)

	omega = eop[0]
	phi = eop[1]
	kappa = eop[2]
	x0 = eop[3]
	y0 = eop[5]
	z0 = eop[4]

	delta = np.array([1,1,1,1,1,1])	

	ii = 0
	while max(abs(delta)) > .00001:
		ii += 1

		mw = np.matrix([[1,0,0],[0,cos(omega),sin(omega)],[0,-sin(omega),cos(omega)]])
		mp = np.matrix([[cos(phi),0,-sin(phi)],[0,1,0],[sin(phi),0,cos(phi)]])
		mk = np.matrix([[cos(kappa),sin(kappa),0],[-sin(kappa),cos(kappa),0],[0,0,1]])

		# m = mk*mp*mw
		m = np.round(mk*mp*mw,6)

		# print("MMM:",m)
		gg = ng * 2

		dx = np.zeros(ng)
		dy = np.zeros(ng)
		dz = np.zeros(ng)
		q = np.zeros(ng)
		r = np.zeros(ng)
		s = np.zeros(ng)
		
		for k in range(0,ng):
			dx[k] = x[k] - x0
			dy[k] = y0 - y[k]
			dz[k] = z[k] - z0
			q[k] = m[2,0]*(x[k] - x0) + m[2,1]*(z[k] - z0) + m[2,2]*(y0 - y[k])
			r[k] = m[0,0]*(x[k] - x0) + m[0,1]*(z[k] - z0) + m[0,2]*(y0 - y[k])
			s[k] = m[1,0]*(x[k] - x0) + m[1,1]*(z[k] - z0) + m[1,2]*(y0 - y[k])

		# print("q:",q)
		# print("r:",r)
		# print("s:",s)
		j = 0
		ff = np.mat(np.zeros((gg,1)))
		for k in range(0,gg,2):
			ff[k,0] = -(q[j]*xp[j]+r[j]*f)/q[j]
			ff[k+1,0] = -((q[j]*yp[j]+s[j]*f)/q[j])
			j += 1

		j = 0
		b = np.mat(np.zeros((gg,6)))
		for k in range(0,gg,2):
			b[k,0] = (xp[j]/q[j])*(-m[2,2]*dz[j]+m[2,1]*dy[j])+(f/q[j])*(-m[0,2]*dz[j]+m[0,1]*dy[j])
			b[k,1] = (xp[j]/q[j])*(dx[j]*cos(phi)+dz[j]*(sin(omega)*sin(phi))+dy[j]*(-sin(phi)*cos(omega)))+(f/q[j])*(dx[j]*(-sin(phi)*cos(kappa))+dz[j]*(sin(omega)*cos(phi)*cos(kappa))+dy[j]*(-cos(omega)*cos(phi)*cos(kappa)))
			b[k,2] = (f/q[j])*(m[1,0]*dx[j]+m[1,1]*dz[j]+m[1,2]*dy[j])
			b[k,3] = -((xp[j]/q[j])*m[2,0] + (f/q[j])*m[0,0])
			b[k,4] = -((xp[j]/q[j])*m[2,1] + (f/q[j])*m[0,1])
			b[k,5] =  ((xp[j]/q[j])*m[2,2] + (f/q[j])*m[0,2])

			b[k+1,0] = (yp[j]/q[j])*(-m[2,2]*dz[j]+m[2,1]*dy[j])+(f/q[j])*(-m[1,2]*dz[j]+m[1,1]*dy[j])
			b[k+1,1] = (yp[j]/q[j])*(dx[j]*cos(phi)+dz[j]*(sin(omega)*sin(phi))+dy[j]*(-sin(phi)*cos(omega)))+(f/q[j])*(dx[j]*(sin(phi)*sin(kappa))+dz[j]*(-sin(omega)*cos(phi)*sin(kappa))+dy[j]*(cos(omega)*cos(phi)*sin(kappa)))
			b[k+1,2] = (f/q[j])*(-m[0,0]*dx[j]-m[0,1]*dz[j]-m[0,2]*dy[j])
			b[k+1,3] = -((yp[j]/q[j])*m[2,0] + (f/q[j])*m[1,0])
			b[k+1,4] = -((yp[j]/q[j])*m[2,1] + (f/q[j])*m[1,1])
			b[k+1,5] =  ((yp[j]/q[j])*m[2,2] + (f/q[j])*m[1,2])
			j += 1

		btb = (b.T*b).I
		btf = b.T * ff
		delta = btb * btf
		v = b*delta - ff
		# D[:,ii] = delta;
		omega = omega + delta[0,0]
		phi = phi + delta[1,0]
		kappa = kappa + delta[2,0]
		x0 = x0 + delta[3,0]
		y0 = y0 + delta[5,0]
		z0 = z0 + delta[4,0]
		# print("ii:",ii,"res:",omega,phi,kappa,x0,y0,z0)
	return omega,phi,kappa,x0,y0,z0

def pre_processing(XYZ,xy):
	x0,y0,f = 2325,1736,3623
	pixel_size = 1.12e-6

	xy = np.array(xy)
	XYZ = np.array(XYZ)
	
	XYZ = XYZ * 0.001
	
	xy[:,0] = xy[:,0] - x0
	xy[:,1] = xy[:,1] - y0
	xy = xy * pixel_size
	f *= pixel_size

	return XYZ,xy,f

def check_distance(a,b,A,B,O,x0,y0,f):
	o = (x0,y0)
	g = ((b[0]-a[0])/2 + a[0],(b[1]-a[1])/2 + a[1])
	G = ((B[0]-A[0])/2 + A[0],(B[1]-A[1])/2 + A[1],(B[2]-A[2])/2 + A[2])
	Oo = f
	og = sqrt(pow(g[0] - o[0],2)+pow(g[1]-o[1],2))
	Og = sqrt(pow(Oo,2)+pow(og,2))
	AB = sqrt(pow(A[0]-B[0],2) + pow(A[1]-B[1],2) + pow(A[2]-B[2],2))
	ab = sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1],2))
	OG = Og/ab *AB
	de = sqrt(pow(O[0]-G[0],2) + pow(O[1]-G[1],2))
	res = sqrt(pow(OG-de,2))
	# print("res:",gamma*0.001)
	return res

def position_estimation(XYZ,xy):
	A = XYZ[0]
	B = XYZ[-1]
	a = xy[0]
	b = xy[-1]
	XYZ,xy,f = pre_processing(XYZ,xy)
	stop = 0
	for omega_init in range(-170,170,20):
		for phi_init in range(-80,80,10):
			eop = [round(d2r(omega_init),5),round(d2r(phi_init),5),round(d2r(0),5),0.0,0.05,0.1]
			omega,phi,kappa,x0,y0,z0 = space_resection(XYZ,xy,eop,f)
			if x0 < 21.0 and x0 > -1.0 and y0 < 4 and y0 > -1.0 and z0 > 0.0:
				# print(omega_in,phi_in)
				stop = 1
				break
		if stop:
			break

	O = (x0*1000,y0*1000,z0*1000)
	res = check_distance(a,b,A,B,O,2325,1736,3623)
	return x0,y0,z0,round(res*0.001,3)




if __name__ == '__main__':
	# DOORS 2-3
	xy = [(1896, 1039), (2971, 1057), (1914, 2137), (2927, 2165), (1932, 3162), (2886, 3197), (166, 1004), (1157, 1029), (147, 2071), (1169, 2092), (164, 3207), (1145, 3205)]
	XYZ = [(2670, 3490, 2160), (3670, 3490, 2160), (2670, 3490, 1080), (3670, 3490, 1080), (2670, 3490, 0), (3670, 3490, 0), (1170, 3490, 2160), (2170, 3490, 2160), (1170, 3490, 1080), (2170, 3490, 1080), (1170, 3490, 0), (2170, 3490, 0)]
	
	# DOORS 6-7
	xy = [(1581, 107), (2730, 381), (1635, 1865), (2690, 1737), (1680, 3338), (2590, 2955), (3195, 326), (3885, 453), (3148, 1659), (3776, 1608), (3092, 2763), (3683, 2586)]
	XYZ = [(13290, 2670, 2160), (14290, 2670, 2160), (13290, 2670, 1080), (14290, 2670, 1080), (13290, 2670, 0), (14290, 2670, 0), (14710, 2670, 2160), (15710, 2670, 2160), (14710, 2670, 1080), (15710, 2670, 1080), (14710, 2670, 0), (15710, 2670, 0)]
	
	# DOORS 4-5
	xy = [(1347, 966), (1766, 975), (2206, 985), (1354, 1466), (1766, 1476), (2199, 1486), (1372, 1978), (1766, 1999), (2203, 2020), (586, 802), (856, 882), (575, 2236), (843, 2120)]
	XYZ = [(0, 130, 2170), (0, 1050, 2170), (0, 1970, 2170), (0, 130, 1085), (0, 1050, 1085), (0, 1970, 1085), (0, 130, 0), (0, 1050, 0), (0, 1970, 0), (2425, 0, 2140), (1550, 0, 2140), (2425, 0, 0), (1550, 0, 0)]
	
	# DOORS 8-9
	xy = [(1324, 926), (1989, 950), (2585, 972), (1328, 1601), (1981, 1622), (2573, 1641), (1329, 2383), (1972, 2384), (2561, 2399)]
	XYZ = [(20550, 480, 2140), (20550, 1400, 2140), (20550, 2320, 2140), (20550, 480, 1070), (20550, 1400, 1070), (20550, 2320, 1070), (20550, 480, 0), (20550, 1400, 0), (20550, 2320, 0)]
	# position_estimation(XYZ,xy)
	x0,y0,f = 2325,1736,3623
	x0,y0,z0,res = position_estimation(XYZ,xy)
	print("SUCCESS!!!!!")
	print("Robot Position    is: X:",round(x0,3),"m , Y:",round(y0,3),"m , Z:",round(z0,3),"m")
	print("Residual =",res)
	# print("Robot Orientation is: ω:",round(r2d(omega),3),"degrees ,φ:",round(r2d(phi),3),"degrees ,κ:",round(r2d(kappa),3),"degrees")	
	# O  = [int(x*1000) for x in O]
	# check_distance(XYZ,xy,O,x0,y0,f)