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

CPTs = [(20,560),(450,560),(840,560),(35,1000),(450,1000),(830,1000),(50,1490),(450,1480),(820,1480)]

COLOURs = [(0,0,100),(100,0,0),(0,100,0)]

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

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

###### CONTROL POINT EXTRACTION FUNCTIONS #######

def Homography(kp1,des1,kp2,des2):
	FLANN_INDEX_KDTREE = 1

	# Matcher
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	i = 0
	for m,n in matches:
		i = i + 1
		if m.distance < 0.75*n.distance:		
			good.append(m)

		
	print("Good matches detected: {}".format(len(good)))
	if len(good) > 0:
		# Homografy creation with RANSAC filtering
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,3.0)
		matchesMask = mask.ravel().tolist()
		sumx = 0
		for x in matchesMask:
			sumx +=  x
		print("Final matches after RANSAC: {}".format(sumx))
		return M,matchesMask,good
	else:
		return [],[],[]

def find_best_homography(ref_paths,test):
	MIN_MATCH_COUNT = 30
	MIN_FILTERED = 10

	num_good = 0
	num_matches = 0
	
	ref_image_path = ""
	best_M = []
	best_Mask = []
	best_good = []

	img2 = cv.imread(test,cv.COLOR_BGR2GRAY)
	kp2,des2 = sift(img2)	
	for ref_path in ref_paths:
		print(ref_path)
		img1 = cv.imread(ref_path,cv.COLOR_BGR2GRAY)
		kp1,des1 = sift(img1)
		
		M,matchesMask,good = Homography(kp1,des1,kp2,des2)
		
		sumx = 0
		for x in matchesMask:
			sumx +=  x

		if len(good) >= MIN_MATCH_COUNT and sumx >= MIN_FILTERED and sumx >= num_matches:
			num_good = len(good)
			num_matches = sumx
			best_M = M
			best_Mask = matchesMask
			best_good = good
			ref_image_path = ref_path
	print(ref_image_path)
	return ref_image_path,best_M,best_Mask,best_good

def sift(img):
	sift = cv.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)
	des = np.float32(des)
	return kp,des

######## VISUALISATION FUNCTIONS #########

def test_sift(ref,test):
	img1 = cv.imread(ref)	# queryImage
	img2 = cv.imread(test) # trainImage
	k1, des1 = sift(img1)
	k2, des2 = sift(img2)
	img1 = cv.drawKeypoints(img1,k1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img2 = cv.drawKeypoints(img2,k2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv.imwrite('../../images/sift/ref_keypoints.jpg',img1)
	cv.imwrite('../../images/sift/test_keypoints.jpg',img2)


def draw_contour(img1,img2,M,src_pts):
	COLOUR = (0,255,0)
	img1 = cv.polylines(img1,[np.int32(src_pts)],True,COLOUR,3,cv.LINE_AA)
	dst = cv.perspectiveTransform(src_pts,M)
	img2 = cv.polylines(img2,[np.int32(dst)],True,COLOUR,3, cv.LINE_AA)


def draw_CPTs(img,CPTs,COLOUR):
	case = 0
	if type(img) is str:
		case = 1
		img = cv.imread(img)
	for CPT in CPTs:
		if type(CPT) is tuple:
			draw_circle(img,tuple(CPT),COLOUR)
		else:
			draw_circle(img,tuple(CPT[0]),COLOUR)
	if case:
		plt.imshow(img[:,:,::-1]),plt.show()

def draw_circle(img,centre,COLOUR):
	# img = cv.imread(img_path)
	THICKNESS = 3
	img = cv.circle(img,centre,50,COLOUR,THICKNESS)
	img = cv.circle(img,centre,3,COLOUR,THICKNESS)


def draw_sift_matching(ref,test,M=[],matchesMask=[],good=[]):

	img1 = cv.imread(ref,cv.COLOR_BGR2GRAY)
	img2 = cv.imread(test,cv.COLOR_BGR2GRAY)

	kp1, des1 = sift(img1)
	kp2, des2 = sift(img2)
	if M == []:
		M,matchesMask,good = Homography(kp1,des1,kp2,des2)
	

	# DRAW CPTS to Both Images
	# pts = np.float32([ [500,170],[500,930],[700,930],[700,170] ]).reshape(-1,1,2)
	pts = np.float32([(500,310),(500,730),(500,1150),(670,370)]).reshape(-1,1,2)
	
	draw_CPTs(img1,pts,COLOURs[0])
	dst = cv.perspectiveTransform(pts,M)
	draw_CPTs(img2,dst,COLOURs[0])

	# draw_contour(img1,img2,M,pts)


	draw_params = dict(matchColor = (0,0,255), 
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	plt.imshow(img3[:,:,::-1]),plt.show()

##### MAIN TESTING FUNCTIONS ######
def main1():
	TEST_IMAGE_PATHS = '../../images/detected_doors/DOOR8.jpg'
	PATH_TO_REF_IMAGES_DIR = "../../images/reference/DOOR8"
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	ref_image_path,M,matchesMask,good = find_best_homography(REF_IMAGE_PATHS,TEST_IMAGE_PATHS)
	# ref_image_path = PATH_TO_REF_IMAGES_DIR
	draw_sift_matching(ref_image_path,TEST_IMAGE_PATHS,M,matchesMask,good)

def main2():
	TEST_IMAGE_PATH = '../../images/detected_doors/DOOR8.jpg'
	REF_IMAGE_PATH = "../../images/reference/DOOR8/DOOR8.5.jpg"
	CPTs = [(500,310),(500,730),(500,1150),(670,370)]
	draw_CPTs(REF_IMAGE_PATH,CPTs,COLOURs[0])

if __name__ == '__main__':
	main1()
