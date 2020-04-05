#!/usr/bin/env python3
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys

import door_detection

import tarfile

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

import cv2 as cv


def isolate_doors(detected_objects):

	(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
		ymin * im_height, ymax * im_height)

	return detected_models

def get_reference_image_path():
	print(sys.path[-1])
	PROJECT_HOME_FOLDER = 	"../../"
	PATH_TO_REF_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/reference"
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	
	PATH_TO_TEST_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/test"
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

	print(TEST_IMAGE_PATHS)
	image = Image.open(TEST_IMAGE_PATHS[0])
	return TEST_IMAGE_PATHS

if __name__ == '__main__':
	TEST_IMAGE_PATHS = get_reference_image_path()

	detected_objects = find_test_doors()

	print(detected_objects)


	# img = cv.imread(TEST_IMAGE_PATHS[0])
	# gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	
	# sift = cv.xfeatures2d.SIFT_create()
	# kp = sift.detect(gray,None)
	
	# img=cv.drawKeypoints(gray,kp,img)
	# cv.imwrite('sift_keypoints.jpg',img)

	# image = Image.open('sift_keypoints.jpg')
	# image.show()

# 	import cv2
# img = cv2.imread("lenna.png")
# crop_img = img[y:y+h, x:x+w]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)

