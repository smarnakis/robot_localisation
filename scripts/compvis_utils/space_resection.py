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

from math import cos
from math import sin
# import math.cos as cos

PI = 3.1415

def main(space_coords,picture_coords,initial_vector,f):
	xa = picture_coords[:,0]
	ya = picture_coords[:,1]
	
	XA = space_coords[:,0]
	YA = space_coords[:,2]
	ZA = space_coords[:,1]

	omega = initial_vector[0]
	phi = initial_vector[1]
	kappa = initial_vector[2]
	X0 = initial_vector[3]
	Y0 = initial_vector[4]
	Z0 = initial_vector[5]

	delta = np.array([1,1,1,1,1,1])

	num_points = len(XA)
	ii = 0
	while max(abs(delta)) > 0.00001:
		ii += 1
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
			dy[i] = Y0 - YA[i]
			dz[i] = ZA[i] - Z0
			q[i] = m[2,0]*(XA[i] - X0) + m[2,1]*(ZA[i] - Z0) + m[2,2]*(Y0 - YA[i])
			r[i] = m[0,0]*(XA[i] - X0) + m[0,1]*(ZA[i] - Z0) + m[0,2]*(Y0 - YA[i])
			s[i] = m[1,0]*(XA[i] - X0) + m[1,1]*(ZA[i] - Z0) + m[1,2]*(Y0 - YA[i])

		double_length = num_points * 2
		j = 0
		L = np.zeros(double_length)
		for k in range(0,double_length,2):
			L[k] = -(q[j]*xa[j]+r[j]*f)/q[j]
			L[k+1] = -(q[j]*ya[j]+s[j]*f)/q[j]
			j += 1
		L = np.matrix(L)
		j = 0
		b = np.zeros((double_length,6))
		for k in range(0,double_length,2):
			b[k,0] = (xa[j]/q[j])*(-m[2,2]*dz[j] + m[2,1]*dy[j]) + (f/q[j])*(-m[0,2]*dz[j] + m[0,1]*dy[j])
			b[k,1] = (xa[j]/q[j])*(dx[j]*cos(phi) + dz[j]*(sin(omega)*sin(phi)) + dy[j]*(-sin(phi)*cos(omega))) + (f/q[j])*(dx[j]*(-sin(phi)*cos(kappa)) + dz[j]*(sin(omega)*cos(phi)*cos(kappa))+dy[j]*(-cos(omega)*cos(phi)*cos(kappa)))
			b[k,2] = (f/q[j])*(m[1,0]*dx[j]+m[1,1]*dz[j]+m[1,2]*dy[j])
			b[k,3] = -((xa[j]/q[j])*m[2,0]+(f/q[j])*m[0,0])
			b[k,4] = -((xa[j]/q[j])*m[2,1]+(f/q[j])*m[0,1])
			b[k,5] = ((xa[j]/q[j])*m[2,2]+(f/q[j])*m[0,2])
			b[k+1,0] = (ya[j]/q[j])*(-m[2,2]*dz[j]+m[2,1]*dy[j])+(f/q[j])*(-m[1,2]*dz[j]+m[1,1]*dy[j])
			b[k+1,1] = (ya[j]/q[j])*(dx[j]*cos(phi)+dz[j]*(sin(omega)*sin(phi))+dy[j]*(-sin(phi)*cos(omega))) + (f/q[j])*(dx[j]*(sin(phi)*sin(kappa))+dz[j]*(-sin(omega)*cos(phi)*sin(kappa))+dy[j]*(cos(omega)*cos(phi)*sin(kappa)))
			b[k+1,2] = (f/q[j])*(-m[0,0]*dx[j]-m[0,1]*dz[j]-m[0,2]*dy[j])
			b[k+1,3] = -((ya[j]/q[j])*m[2,0]+(f/q[j])*m[1,0])
			b[k+1,4] = -((ya[j]/q[j])*m[2,1]+(f/q[j])*m[1,1])
			b[k+1,5] = ((ya[j]/q[j])*m[2,2]+(f/q[j])*m[1,2])
			j += 1
		b = np.matrix(b)
		if ii == 1:
			print(b)
		# print(np.transpose(b))
		# print(b)
		# ml = np.matmul(np.transpose(b)*b)
		ml = np.transpose(b)*b
		btb = inv(ml)
		btf = np.transpose(b)*np.transpose(L)
		delta = btb*btf
		omega = omega + delta[0]
		phi = phi + delta[1]
		kappa = kappa + delta[2]
		X0 = X0 + delta[3]
		Y0 = Y0 + delta[4]
		Z0 = Z0 + delta[5]
		if ii > 100:
			break
	if ii > 100:
		return ('E','E','E','E','E','E')
	else:
		return (X0,Y0,Z0,omega,phi,kappa)


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

def main1():
	x = np.matrix(1,2)
	y = np.matrix([[1,2,3],
								[4,5,6]])
	print(x)

if __name__ == '__main__':
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
	initial_coords = [X0,Y0,Z0,omega,phi,kappa]
	# print(image_coords)
	# print(space_coords)
	# print(initial_coords)
	# print(image_coords)
	camera_coords = main(space_coords,image_coords,initial_coords,f)
	print(camera_coords)
	# main1()