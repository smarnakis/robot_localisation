#!/usr/bin/env python3
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys

import tarfile

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from IPython.display import display

import cv2 as cv


def get_reference_image_path():
	print(sys.path[-1])
	PROJECT_HOME_FOLDER = 	"../../"
	PATH_TO_REF_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/reference"
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	print(REF_IMAGE_PATHS)	
	PATH_TO_TEST_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/detected_doors"
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

	print(TEST_IMAGE_PATHS)
	#image = Image.open(TEST_IMAGE_PATHS[0])
	return TEST_IMAGE_PATHS,REF_IMAGE_PATHS


def Homography(kp1,des1,kp2,des2):
	MIN_MATCH_COUNT = 30
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	#print(matches)
	# print(len(des1))
	# print(len(des2))
	# store all the good matches as per Lowe's ratio test.
	good = []
	#print(matches[0][0].shape)
	i = 0
	for m,n in matches:
		i = i + 1
		if m.distance < 0.75*n.distance:
			print(i)
			print("m.distance = {}, n.distance = {}".format(m.distance,n.distance))
			print("m.queryIdx = {}, n.queryIdx = {}".format(m.queryIdx,n.queryIdx))
			print("m.trainIdx = {}, n.trainIdx = {}".format(m.trainIdx,n.trainIdx))			
			good.append(m)
	if len(good)>MIN_MATCH_COUNT:
		print(len(good))
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,3.0)
		x = 0
		for i in mask:
			x = x + i
		print(x)
		# print(len_mask)
		# M, mask = cv.findHomography(src_pts, dst_pts,0,5.0)
		matchesMask = mask.ravel().tolist()
	return M,matchesMask,good

def sift(img):
	sift = cv.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)
	des = np.float32(des)
	return kp,des

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def test_sift_matching(ref,test):

	img1 = cv.imread(ref,cv.COLOR_BGR2GRAY)
	img2 = cv.imread(test,cv.COLOR_BGR2GRAY)

	kp1, des1 = sift(img1)
	kp2, des2 = sift(img2)
	
	M,matchesMask,good = Homography(kp1,des1,kp2,des2)
	
	pts = np.float32([ [20,170],[20,930],[400,930],[400,170] ]).reshape(-1,1,2)
	img1 = cv.polylines(img1,[np.int32(pts)],True,(0,255,0),3,cv.LINE_AA)
	dst = cv.perspectiveTransform(pts,M)
	img2 = cv.polylines(img2,[np.int32(dst)],True,(0,255,0),3, cv.LINE_AA)
	draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	plt.imshow(img3, 'gray'),plt.show()


def crop_pil(im_path):
	pil = Image.open(im_path)
	width, height = pil.size 
	left = width / 5
	top = height / 4 - 50
	right = 4 * width / 5 - 50
	bottom = 3 * height / 4 -50
	res = pil.crop((left,top,right,bottom))
	#res.show()
	res.save('../../images/reference/DOOR5.3.jpg')


def test_orb(ref,test):
	img1 = cv.imread(ref)	# queryImage
	img2 = cv.imread(test) # trainImage
	k1, des1 = orb(img1)
	k2, des2 = orb(img2)
	img1 = cv.drawKeypoints(img1,k1,img1)
	img2 = cv.drawKeypoints(img2,k2,img2)
	cv.imwrite('../../images/sift/ref_keypoints.jpg',img1)
	cv.imwrite('../../images/sift/test_keypoints.jpg',img2)

def test_sift(ref,test):
	img1 = cv.imread(ref)	# queryImage
	img2 = cv.imread(test) # trainImage
	k1, des1 = sift(img1)
	k2, des2 = sift(img2)
	img1 = cv.drawKeypoints(img1,k1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img2 = cv.drawKeypoints(img2,k2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv.imwrite('../../images/sift/ref_keypoints.jpg',img1)
	cv.imwrite('../../images/sift/test_keypoints.jpg',img2)

if __name__ == '__main__':
	TEST_IMAGE_PATHS = '../../images/detected_doors/DOOR4.jpg'
	REF_IMAGE_PATHS ='../../images/reference/DOOR4.jpg'

	#main()
	# crop_pil(REF_IMAGE_PATHS)
	test_sift_matching(REF_IMAGE_PATHS,TEST_IMAGE_PATHS)
	# test_orb(REF_IMAGE_PATHS,TEST_IMAGE_PATHS)