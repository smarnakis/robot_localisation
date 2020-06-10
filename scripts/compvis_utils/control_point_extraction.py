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
# from test import examine_sift_error

COLOURs = [(0,0,100),(0,100,0),(0,100,100),(100,0,0),
						(100,0,100),(100,100,0),(100,50,100),
						(100,100,200),(100,100,50)]
def crop_pil(im_path):
	pil = Image.open(im_path)
	width, height = pil.size 
	left = width / 8
	top = 0
	right =2* width / 3
	bottom = height
	res = pil.crop((left,top,right,bottom))
	res.show()
	# res.save('../../images/reference/DOOR6/DOOR6.5.jpg')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

###### CONTROL POINT EXTRACTION FUNCTIONS #######

#<-------- HOMOGRAPHY RELATED FUNCTIONS -------># 
def Homography(kp1,des1,kp2,des2):
	# Usage: Given the key points and desctiptors of a scr and dst image
	# it computes the homography matrix betwenn the two 
	# images, using Lowe's ratio filtering and RANSAC filtering.
	# Inputs: kp1: Source image keypoints
	# 				des1: Source image descriptors
	# 				kp2: Destination image keypoints
	# 				des2: Destination image descriptors
	# Outputs: 	M: Homography matrix
	# 					matchesMask: Final inliers after RANSAC and Lowe's
	# 					good: Inliers of Lowe's filtering
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
	MIN_MATCH_COUNT = 27
	MIN_FILTERED = 9

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

def find_best_homo_pair(REF_PATHS,img2):
	# USAGE: Given a test image block and the corresponding reference
	# images, it finds the reference image that can be better matched
	# with the test image block.
	# INPUTS: REF_PATHS: the paths to the reference images that correspond to the test image block
	#					img2: the test image block in np.array format [width,length,channels(BGR)]
	# OUTPUS: BEST_REF_PATH: the path to the best reference image
	#					best_M: The homography matrix between the test image block and the best ref image
	MIN_MATCH_COUNT = 29
	MIN_FILTERED = 9

	num_good = 0
	num_matches = 0
	
	BEST_REF_PATH = ""
	best_M = []

	kp2,des2 = sift(img2)	
	for REF_PATH in REF_PATHS:
		print(REF_PATH)
		img1 = cv.imread(REF_PATH,cv.COLOR_BGR2GRAY)
		kp1,des1 = sift(img1)
		
		M,matchesMask,good = Homography(kp1,des1,kp2,des2)
		
		sumx = 0
		for x in matchesMask:
			sumx +=  x

		if len(good) >= MIN_MATCH_COUNT and sumx >= MIN_FILTERED and sumx >= num_matches:
			num_good = len(good)
			num_matches = sumx
			best_M = M
			BEST_REF_PATH = REF_PATH
	print("The best ref is: " + BEST_REF_PATH)

	return BEST_REF_PATH,best_M

def sift(img):
	# USAGE: Performs the SIFT algorithm
	# INPUTS: img: An image in np.array format [width,lenght,channels(BGR)]
	# OUTPUTS: kp: The detected keypoints
	# 				 des: The discriptors of the detected keypoints
	sift = cv.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)
	des = np.float32(des)
	return kp,des

#<---------- CONTROL POINTS PROCESSING FUNCTIONS ---------># 
def bring_ref_CPTS(tags):
	# USAGE: Given the desired reference image tags, it retrieves the
	# corresponding CPTS of the reference images from the database
	# INPUTS: tags: String list with the image tags
	# OUTPUTS: data: tuple of format: 
	# (String:ref_image_tag,list of tuples:Image CPT coords,list of tuples: Space CPT coords) 
	DATABASE_PATH = "../../images/reference/CPTS.txt"
	f = open(DATABASE_PATH,"rt")
	lines = f.readlines()
	data = []
	i = 0
	print(tags)
	last_tag = tags[-1]
	for line in lines:
		while(tags[i]==""):
			print("i=",i)
			data.append(())
			if i < len(tags)-1:
				i += 1
			else:
				i += 1
				break
		if i == len(tags):
			break
		line = line.split(" | ")
		# print(line[0],tags[i])
		if tags[i] == line[0]:
			i += 1
			print(i)
			data.append((line[0],ast.literal_eval(line[1]),ast.literal_eval(line[2])))
		if tags[i-1] == last_tag and i!=0:
			break
	print(data)
	f.close()
	return data

def trans_relative_test_CPTS(ref_data,Ms):
	# USAGE: Uses the Homography transform to 
	rel_CPTS = []
	space_coords = []
	for i in range(len(Ms)):
		if Ms[i] != []:
			ref_CPTS = np.float32(ref_data[i][1]).reshape(-1,1,2)
			space_coords.append(ref_data[i][2])
			CPTS = cv.perspectiveTransform(ref_CPTS,Ms[i])
			rel_CPTS.append(CPTS)
		else:
			rel_CPTS.append([])
	return rel_CPTS,space_coords
	
def abs_test_CPTS(relative_test_CPTS,origins):
	test_CPTS = []
	test_CPTS_vis = []
	for i,block in enumerate(relative_test_CPTS):
		if block != []:
			x_or = origins[i][0]
			y_or = origins[i][1]
			vis_block = []
			for CPT in block:
				# print(CPT)
				test_CPTS.append((int(x_or+CPT[0][0]),int(y_or+CPT[0][1])))
				vis_block.append((int(x_or+CPT[0][0]),int(y_or+CPT[0][1])))
			test_CPTS_vis.append(vis_block)
	return test_CPTS,test_CPTS_vis

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

def draw_test_CPTS(img_path,vis_CPTS):
	img = cv.imread(img_path)
	for i,CPTS in enumerate(vis_CPTS):
		for CPT in CPTS:
			draw_circle(img,tuple(CPT),COLOURs[i])
	plt.imshow(img[:,:,::-1]),plt.show()

def draw_CPTs(img,CPTs,COLOUR):
	case = 0
	if type(img) is str:
		case = 1
		img = cv.imread(img)
	for CPT in CPTs:
		if type(CPT) is tuple  or len(CPTs[0])==2:
			draw_circle(img,tuple(CPT),COLOUR)
		else:
			draw_circle(img,tuple(CPT[0]),COLOUR)
	if case:
		plt.imshow(img[:,:,::-1]),plt.show()

def draw_circle(img,centre,COLOUR):
	# img = cv.imread(img_path)
	THICKNESS = 10
	img = cv.circle(img,centre,50,COLOUR,THICKNESS)
	img = cv.circle(img,centre,3,COLOUR,THICKNESS)


def draw_sift_matching(img1,img2,pts=[],M=[],matchesMask=[],good=[]):

	if type(img1) is str:
		img1 = cv.imread(img1,cv.COLOR_BGR2GRAY)
	if type(img2) is str:
		img2 = cv.imread(img2,cv.COLOR_BGR2GRAY)

	kp1, des1 = sift(img1)
	kp2, des2 = sift(img2)
	if M == []:
		M,matchesMask,good = Homography(kp1,des1,kp2,des2)
	

	# DRAW CPTS to Both Images
	# pts = np.float32([ [500,170],[500,930],[700,930],[700,170] ]).reshape(-1,1,2)
	if pts == []:
		pts = np.float32([(10,30),(430,30),(870,30),(30,515),(430,515),(850,515),(60,980),(430,990),(840,1000)]).reshape(-1,1,2)
	else:
		pts = np.float32(pts).reshape(-1,1,2)
	# draw_CPTs(img1,pts,COLOURs[0])
	print("SIFT M:")
	print(M)
	dst = cv.perspectiveTransform(pts,M)
	# draw_CPTs(img2,dst,COLOURs[0])

	# draw_contour(img1,img2,M,pts)


	draw_params = dict(matchColor = (0,0,255), 
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	plt.imshow(img3[:,:,::-1]),plt.show()
	return dst


##### MAIN TESTING FUNCTIONS ######
def check_homography():
	# ymin = 0.33355704
	# xmin = 0.72840554
	TEST_IMAGE_PATHS = '../../images/detected_doors/DOOR4.jpg'
	PATH_TO_REF_IMAGES_DIR = "../../images/reference/DOOR4"
	# TEST_IMAGE_PATH = '../../images/test/TEST7.jpg'
	# test_img = cv.imread(TEST_IMAGE_PATH,cv.COLOR_BGR2GRAY)
	# length,width,channels = test_img.shape
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	ref_image_path,M,matchesMask,good = find_best_homography(REF_IMAGE_PATHS,TEST_IMAGE_PATHS)
	dst = draw_sift_matching(ref_image_path,TEST_IMAGE_PATHS,M=M,matchesMask=matchesMask,good=good)
	# print(dst)
	# print(ymin*length)
	# print(xmin*width)
	# for CPT in dst:
	# 	CPT[0][0] = int(CPT[0][0] + xmin*width)
	# 	CPT[0][1] = int(CPT[0][1] + ymin*length)
	draw_CPTs(TEST_IMAGE_PATHS,dst,COLOURs[0])

def check_CPTS():
	TEST_IMAGE_PATH = '../../images/detected_doors/DOOR6.jpg'
	REF_IMAGE_PATH = "../../images/reference/DOOR3/DOOR3.2.jpg"
	CPTs = [(380,290),(650,140),(375,685),(640,690),(370,1080),(640,1250)]
	draw_CPTs(REF_IMAGE_PATH,CPTs,COLOURs[0])
	draw_sift_matching(REF_IMAGE_PATH,TEST_IMAGE_PATH,CPTs)


def main3():
	ymin = 0.33355704
	xmin = 0.72840554
	TEST_IMAGE_PATH = '../../images/test/TEST1.jpg'
	test_img = cv.imread(TEST_IMAGE_PATH,cv.COLOR_BGR2GRAY)
	length,width,channels = test_img.shape
	CPTs = [[500,600],[500,1000],[1000,1000],[1500,1000]]
	for CPT in CPTs:
		CPT[0] = int(CPT[0] + ymin*length)
		CPT[1] = int(CPT[1] + xmin*width)
	draw_CPTs(test_img,CPTs,COLOURs[0])
	plt.imshow(test_img[:,:,::-1]),plt.show()


def algorithm1(test_image_blocks):
	tags = []
	Ms = []
	origins = []
	PATH_TO_TEST_IMAGE = '../../images/test'
	TEST_IMAGE_PATH = os.path.join(PATH_TO_TEST_IMAGE,os.listdir(PATH_TO_TEST_IMAGE)[0])
	for block in test_image_blocks:
		PATH_TO_REF_IMAGES_DIR = "../../images/reference/DOOR" + str(block[2])
		draw_tag = "../../images/detected_doors/DOOR" + str(block[2]) + ".jpg"
		# LOAD TEST IMG BLOCK
		test_img_block = cv.imread(draw_tag,cv.COLOR_BGR2GRAY)
		REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
		# test_img_block = block[0]
		origins.append(block[1])
		ref_image_path,best_M = find_best_homo_pair(REF_IMAGE_PATHS,test_img_block)
		Ms.append(best_M)
		if best_M != []:
			# print("ALGO1")
			# print(best_M)
			tags.append(ref_image_path.split('/')[-1])
			pts = [(100,200),(540,200),(1010,200),(100,715),(540,715),(1000,715),(100,1260),(540,1260),(1000,1260)]
			draw_sift_matching(ref_image_path,block[0],pts)
		else:
			tags.append("")

	ref_data = bring_ref_CPTS(tags)
	relative_test_CPTS,space_coords = trans_relative_test_CPTS(ref_data,Ms)
	# print("DOOR4 BLOCK IMAGE CPTS:",relative_test_CPTS[0])
	space_coordinates = []
	for block in space_coords:
		space_coordinates = space_coordinates + block
	# print("XYZ=")
	# print(space_coordinates)
	test_CPTS,test_CPTS_vis = abs_test_CPTS(relative_test_CPTS,origins)
	# print("xy=")
	# print(test_CPTS)
	draw_test_CPTS(TEST_IMAGE_PATH,test_CPTS_vis)
	return space_coordinates,test_CPTS


def testing_function(test_image_blocks):
	tags = []
	Ms = []
	origins = []
	PATH_TO_TEST_IMAGE = '../../images/test'
	TEST_IMAGE_PATH = os.path.join(PATH_TO_TEST_IMAGE,os.listdir(PATH_TO_TEST_IMAGE)[0])
	for block in test_image_blocks:
		PATH_TO_REF_IMAGES_DIR = "../../images/reference/DOOR" + str(block[2])
		draw_tag = "../../images/detected_doors/DOOR" + str(block[2]) + ".jpg"
		# LOAD TEST IMG BLOCK
		test_img_block1 = cv.imread(draw_tag,cv.COLOR_BGR2GRAY)
		REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
		test_img_block = block[0]
		if (test_img_block == test_img_block1).all():
			print("SAME MATRICES")
		origins.append(block[1])
		ref_image_path,best_M = find_best_homo_pair(REF_IMAGE_PATHS,test_img_block)
		draw_sift_matching(ref_image_path,test_img_block1)
		draw_sift_matching(ref_image_path,test_img_block)

def test_cropped_images():
	test_cropped,origin = examine_sift_error()
	PATH_TO_REF_IMAGES_DIR = "../../images/reference/DOOR4/DOOR4.4.jpg"

	draw_tag = "../../images/detected_doors/DOOR4.jpg"
	read_cropped = cv.imread(draw_tag,cv.COLOR_BGR2GRAY)
	draw_sift_matching(PATH_TO_REF_IMAGES_DIR,test_cropped)
	draw_sift_matching(PATH_TO_REF_IMAGES_DIR,read_cropped)
	draw_sift_matching(read_cropped,test_cropped)




if __name__ == '__main__':
	# check_homography()
	check_CPTS()
	tags = ['DOOR7.3.jpg', '']
	# test_cropped_images()
	# bring_ref_CPTS(tags)
