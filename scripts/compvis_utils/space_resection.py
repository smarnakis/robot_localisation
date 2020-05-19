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

def main(space_coords,picture_coords,initial_vector,f):
	xa = picture_coords(:,0)
	ya = picture_coords(:,1)
	
	XA = space_coords(:,0)
	YA = space_coords(:,1)
	ZA = space_coords(:,2)

	omega = initial_vector(0)
	phi = initial_vector(1)
	kappa = initial_vector(2)
	X0 = initial_vector(3)
	Y0 = initial_vector(4)
	Z0 = initial_vector(5)

	delta = np.array([1,1,1,1,1,1])

	num_points = len(XA)

	while max(abs(delta)) > 0.00001:
		mw = np.matrix([[1,0,0],[0,cos(omega),sin(omega)],[0,-sin(omega),cos(omega)]])
		mp = np.matrix([[cos(phi),0,-sin(phi)],[0,1,0],[sin(phi),0,cos(phi)]])
		mw = np.matrix([[cos(kappa),sin(kappa),0],[-sin(kappa),cos(kappa),0],[0,0,1]])

		M = np.matmul(mk*mp*mw)

		dx = np.zeros(num_points)
		dy = np.zeros(num_points)
		dz = np.zeros(num_points)
		q = np.zeros(num_points)
		r = np.zeros(num_points)
		s = np.zeros(num_points)
		for i in range(num_points):
			dx(i) = XA(i) - X0
			dy(i) = Y0 - YA(i)
			dz(i) = ZA(i) - Z0
			q(i) = M(3,1)*(XA(i) - X0) + M(3,2)*(ZA(i) - Z0) + M(3,3)*(Y0 - YA(i))
			r(i) = M(1,1)*(XA(i) - X0) + M(1,2)*(ZA(i) - Z0) + M(1,3)*(Y0 - YA(i))
			s(i) = M(2,1)*(XA(i) - X0) + M(2,2)*(ZA(i) - Z0) + M(2,3)*(Y0 - YA(i))

		double_length = num_points * 2
		j = 0
		L = np.zeros(double_length)
		for k in range(0,double_length,2):
			L(k) = -(q(j)*xa(j)+r(j)*f)/q(j)
			L(k+1) = -(q(j)*ya(j)+s(j)*f)/q(j)
			j += 1

		j = 0
		b = np.zeros(double_length,6)
		for k in range(0,double_length,2):
			b(k,0) = (xa(j)/q(j))*(-m(3,3)*dz(j) + m(3,2)*dy(j)) + (f/q(j))*(-m(1,3)*dz(j) + m(1,2)*dy(j))
			b(k,1) = (xa(j)/q(j))*(dx(j)*cos(phi) + dz(j)*(sin(omega)*sin(phi)) + dy(j)*(-sin(phi)*cos(omega))) + (f/q(j))*(dx(j)*(-sin(phi)*cos(kappa)) + dz(j)*(sin(omega)*cos(phi)*cos(kappa))+dy(j)*(-cos(omega)*cos(phi)*cos(kappa)))
	    b(k,2)=(f/q(j))*(m(2,1)*dx(j)+m(2,2)*dz(j)+m(2,3)*dy(j))
	    b(k,3)=-((xa(j)/q(j))*m(3,1)+(f/q(j))*m(1,1))
	    b(k,4)=-((xa(j)/q(j))*m(3,2)+(f/q(j))*m(1,2))
	    b(k,5)= ((xa(j)/q(j))*m(3,3)+(f/q(j))*m(1,3))
	    b(k+1,0)=(ya(j)/q(j))*(-m(3,3)*dz(j)+m(3,2)*dy(j))+(f/q(j))*(-m(2,3)*dz(j)+m(2,2)*dy(j))
	    b(k+1,1)=(ya(j)/q(j))*(dx(j)*cos(phi)+dz(j)*(sin(omega)*sin(phi))+dy(j)*(-sin(phi)*cos(omega))) + (f/q(j))*(dx(j)*(sin(phi)*sin(kappa))+dz(j)*(-sin(omega)*cos(phi)*sin(kappa))+dy(j)*(cos(omega)*cos(phi)*sin(kappa)))
	    b(k+1,2)=(f/q(j))*(-m(1,1)*dx(j)-m(1,2)*dz(j)-m(1,3)*dy(j))
	    b(k+1,3)=-((ya(j)/q(j))*m(3,1)+(f/q(j))*m(2,1))
	    b(k+1,4)=-((ya(j)/q(j))*m(3,2)+(f/q(j))*m(2,2))
	    b(k+1,5)= ((ya(j)/q(j))*m(3,3)+(f/q(j))*m(2,3))
	    j += 1

	  btb = inv(np.matmul(np.transpose(b)*b))
		btf = np.matmul(np.transpose(b)*ff)
		delta = np.matmul(btb*btf)
		omega = omega + delta(0)
		phi = phi + delta(1)
		kappa = kappa + delta(2)
		X0 = X0 + delta(3)
		Y0 = Y0 + delta(4)
		Z0 = Z0 + delta(5)




if __name__ == '__main__':
	main()