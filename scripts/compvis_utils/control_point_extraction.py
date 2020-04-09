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


def Homography(img, prev_img):


    orb = cv2.ORB_create()
    kpt1, des1 = orb.detectAndCompute(prev_img, None)
    kpt2, des2 = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M 

def sift(img):
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	
	sift = cv.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	
	img = cv.drawKeypoints(gray,kp,img)
	

	#image = Image.open('sift_keypoints.jpg')
	#image.show()	

	return img


if __name__ == '__main__':
	TEST_IMAGE_PATHS,REF_IMAGE_PATHS = get_reference_image_path()

	test_img_array = cv.imread(TEST_IMAGE_PATHS[0])
	ref_img_array = cv.imread(REF_IMAGE_PATHS[0])

	test_features_img = sift(test_img_array)
	cv.imwrite('/home/smarn/thesis/robot_localisation/images/detected_doors/test_features_img.jpg',test_features_img)
	ref_features_img = sift(ref_img_array)
	cv.imwrite('/home/smarn/thesis/robot_localisation/images/detected_doors/ref_features_img.jpg',ref_features_img)




# 	import cv2
# img = cv2.imread("lenna.png")
# crop_img = img[y:y+h, x:x+w]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)

