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


def Homography(image_pair):
	kpt1, des1 = sift(image_pair[1])
	kpt2, des2 = sift(image_pair[0])
	bf = cv.BFMatcher()
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
	return M,mask

def sift(img):
	# Input: img in np-array format
	# Output: keypoints and descriptors of image
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	sift = cv.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	#img = cv.drawKeypoints(gray,kp,img)
	#image = Image.open('sift_keypoints.jpg')
	#image.show()	
	return kp,des

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def draw_control_points(image,point):
	# image_pil = Image.fromarray(image)
	# draw = ImageDraw.Draw(image_pil)
	# im_width, im_height = image_pil.size
	# xmin = im_width*0.3
	# xmax = xmin+10
	# ymin = im_height*0.3
	# ymax = ymin+10
	# draw.ellipse([(xmin,ymin),(xmax,ymax)])
	im_test = cv.polylines(image,[np.int32(point)],True,255,3, cv.LINE_AA)
	cv.imwrite('../../images/vis_utils_test/DESTINATION.jpg',im_test)
	# draw.point([(point[0],point[1])])
	# image_pil.save('../../images/vis_utils_test/test.jpg')

def test(ref,test):
	MIN_MATCH_COUNT = 8
	img1 = cv.imread(ref,0)          # queryImage
	img2 = cv.imread(test,0) # trainImage
	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv.perspectiveTransform(pts,M)
		img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	plt.imshow(img3, 'gray'),plt.show()

def main():
	TEST_IMAGE_PATHS,REF_IMAGE_PATHS = get_reference_image_path()
	TEST_IMAGE_PATHS = '../../images/detected_doors/DOOR4.jpg'
	REF_IMAGE_PATHS ='../../images/reference/DOOR4.jpg'
	# image_pil = Image.open(REF_IMAGE_PATHS)
	# image_pil_test = Image.open(TEST_IMAGE_PATHS)
	# ref_img_array_ = load_image_into_numpy_array(image_pil)
	# ref_img_array = ref_img_array_[:,:,::-1].copy()
	
	# test_img_array_ = load_image_into_numpy_array(image_pil_test)
	# test_img_array = test_img_array_[:,:,::-1].copy()

	ref_img_array = cv.imread(REF_IMAGE_PATHS)
	test_img_array = cv.imread(TEST_IMAGE_PATHS)
	M,mask = Homography((test_img_array,ref_img_array))
	print(M)
	im_height, im_width,depth = ref_img_array.shape
	print(im_height,im_width)
	# xmin = im_width*0.3
	# xmax = xmin+100
	# ymin = im_height*0.3
	# ymax = ymin+100	
	xmin = 0
	xmax = im_width-1
	ymin = 0
	ymax = im_height-1
	CTPref = np.float32([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]]).reshape(-1,1,2)
	CTPtest = cv.perspectiveTransform(CTPref,M)
	im_ref = cv.polylines(ref_img_array,[np.int32(CTPref)],True,255,3, cv.LINE_AA)
	cv.imwrite('../../images/vis_utils_test/SOURCE.jpg',im_ref)
	print(CTPref)
	print("------")
	print(CTPtest)
	draw_control_points(test_img_array,CTPtest)

def crop_pil(im_path):
	pil = Image.open(im_path)
	width, height = pil.size 
	left = 0
	top = height / 6 - 40
	right = 5 * width / 6
	bottom = 5 * height / 6 + 150
	res = pil.crop((left,top,right,bottom))
	#res.show()
	res.save('../../images/reference/DOOR7.jpg')

if __name__ == '__main__':
	TEST_IMAGE_PATHS = '../../images/detected_doors/DOOR8.jpg'
	REF_IMAGE_PATHS ='../../images/reference/DOOR8.jpg'

	#main()
	# crop_pil(REF_IMAGE_PATHS)
	test(REF_IMAGE_PATHS,TEST_IMAGE_PATHS)