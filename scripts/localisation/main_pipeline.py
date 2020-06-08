import sys
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2 as cv

sys.path.append('..')
from detection.door_detection import find_test_doors
from compvis_utils.control_point_extraction import algorithm1,testing_function
from compvis_utils.space_resection import position_estimation
matplotlib.use('TkAgg')


def get_ref_test_image_path():
	PROJECT_HOME_FOLDER = 	"../../"
	PATH_TO_REF_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/reference"
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	# print(REF_IMAGE_PATHS)	
	PATH_TO_TEST_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/test"
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

	# print(TEST_IMAGE_PATHS)
	#image = Image.open(TEST_IMAGE_PATHS[0])
	return PATH_TO_REF_IMAGES_DIR,PATH_TO_TEST_IMAGES_DIR


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def exclude_missdetections(boxes,scores,classes,num_detections):
	blocks = [(),(),(),(),(),(),(),(),()]
	num_detected = len(classes)
	important_blocks = []
	tmp_blocks = []
	tags = [1,1,1,1,1,1,1,1,1]
	# Append blocks of each category with best score
	for i in range(num_detected):
		if blocks[classes[i]-1] == ():
			blocks[classes[i]-1] = (boxes[i],scores[i],classes[i])
		elif scores[i] > blocks[classes[i]-1][1]:
			blocks[classes[i]-1] = (boxes[i],scores[i],classes[i])
	
	# Keep only blocks with score > 0.6
	for i,block in enumerate(blocks):
		if block[1] < 0.6:
			if block[2] == 1:
				tags[0] = 0
			if block[2] == 2:
				tags[1] = 0
			if block[2] == 3:
				tags[2] = 0
			if block[2] == 4:
				tags[3] = 0
			if block[2] == 5:
				tags[4] = 0
			if block[2] == 6:
				tags[5] = 0
			if block[2] == 7:
				tags[6] = 0
			if block[2] == 8:
				tags[7] = 0
			if block[2] == 9:
				tags[8] = 0
	print(blocks)
	for i in range(9):
		if tags[i] == 1:
			if i == 3:
				if tags[8] == 0 or blocks[i][1] > blocks[8][1]:
					important_blocks.append(blocks[i])
			elif i == 4:
				if tags[8] == 0 or blocks[i][1] > blocks[8][1]:
					important_blocks.append(blocks[i])
			elif i == 5:
				if tags[8] == 1 and tags[6] == 1:
					important_blocks.append((blocks[i][0],blocks[i][1],6))
				elif tags[8] == 1 and tags[6] == 0:
					# important_blocks.append((blocks[i][0],blocks[i][1],7))
					print("do nothing")
				else:
					important_blocks.append((blocks[i][0],blocks[i][1],7))
			else:
				important_blocks.append(blocks[i])
	# print(important_blocks)

	return important_blocks

def chop_block_PIL(or_image,block_bounds):
	# Usage: Crops an image to seperate image blocks
	# Inputs: 1) Original image in np.array format (width,length,channels) RGB
	# 				2) Image block boundaries in image size percentages
	#	
	# Outputs:1) Cropped image np.array format (width,length,channels) BGR
	# 				2) Upper-left pixel coords of cropped image in original
	image_pil = Image.fromarray(or_image)
	im_width,im_height = image_pil.size
	print("PIL WIDTH:",im_width)
	ymin = block_bounds[0]
	xmin = block_bounds[1]
	ymax = block_bounds[2]	
	xmax = block_bounds[3]	
	left, bottom, right, top = xmin * im_width, ymax * im_height, xmax * im_width, ymin * im_height
	print("PIL BOUNDS: l,b,r,t =",left, bottom, right, top)
	chopped_pil = image_pil.crop((int(left), int(top), int(right), int(bottom)))
	cropped_im = load_image_into_numpy_array(chopped_pil)
	origin = (left,top)
	
	return cropped_im[:,:,::-1],origin

def chop_block_CV(or_image,block_bounds):
	# Usage: Crops an image to seperate image blocks (OPENCV)
	# Inputs: 1) Original image in np.array format (length,width,channels) BGR
	# 				2) Image block boundaries in image size percentages
	#	
	# Outputs:1) Cropped image np.array format (length,width,channels) BGR
	# 				2) Upper-left pixel coords of cropped image in original
	im_height,im_width,channels = or_image.shape
	ymin = block_bounds[0]
	xmin = block_bounds[1]
	ymax = block_bounds[2]	
	xmax = block_bounds[3]	
	left, bottom, right, top = int(xmin * im_width), int(ymax * im_height), int(xmax * im_width), int(ymin * im_height)
	cropped_im = or_image[top:bottom,left:right,:]
	# plt.imshow(cropped_im[:,:,::-1]),plt.show()
	origin = (left,top)
	print(origin)
	return cropped_im,origin


def save_image_parts(image_blocks,method):
	if method == "pil":
		for block in image_blocks:
			tmp_pil = Image.fromarray(block[0][:,:,::-1])
			tmp_label = block[2]
			image_name = 'DOOR' + str(tmp_label) + '.jpg'
			tmp_pil.save('../../images/detected_doors/'+image_name)
	else:
		for block in image_blocks:
			img = block[0]
			tmp_label = block[2]
			image_name = 'DOOR' + str(tmp_label) + '.jpg'
			cv.imwrite('../../images/detected_doors/'+image_name,img)




def seperate_blocks(or_image,imp_bl):
	# Returns all detected doors in tuple of form:
	# image_block = (door-image-part in np-array,score,label)
	image_blocks = []
	for block in imp_bl:
		# print('DOOR')
		# print(block[2])
		# print(block[1])
		cropped_im,origin = chop_block_CV(or_image,block[0])
		image_blocks.append((cropped_im,origin,block[2]))
	return image_blocks

def match_with_reference(image_blocks,PATH_TO_REF_IMAGES_DIR):
	"""
	Input: image_block-type
	Output: List with test-reference picture pairs in np-arrays.
	"""
	test_ref_pairs = []
	for block in image_blocks:
		label = block[2]
		IMAGE_LABEL = 'DOOR' + str(label) + '.jpg'
		ref_image = os.path.join(PATH_TO_REF_IMAGES_DIR,IMAGE_LABEL)
		ref_im_pil = Image.open(ref_image)
		ref_array = load_image_into_numpy_array(ref_im_pil)
		test_ref_pairs.append((block[0],ref_array))
	return test_ref_pairs




def main():
	# Bring test image and image blocks
	PATH_TO_REF_IMAGES_DIR,PATH_TO_TEST_IMAGES_DIR = get_ref_test_image_path()
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
	print("#------------------------------------------------#")
	print("#~~~~~~~~~~Robot localisation algorithm!~~~~~~~~~#")
	print()
	print("--------------------TENSORFOW WARN ---------------------------------")
	_,boxes,scores,classes,num_detections = find_test_doors(PATH_TO_TEST_IMAGES_DIR)
	print("--------------------------------------------------------------------")
	print("# 1st stage ignition: Static Object Detection  #")
	print("# Calculating...")	
	important_blocks = exclude_missdetections(boxes,scores,classes,num_detections)
	image = cv.imread(TEST_IMAGE_PATHS[0],cv.COLOR_BGR2GRAY)
	print("TEST IMAGE TYPE:",image.dtype)
	image_blocks = seperate_blocks(image,important_blocks)
	print("# First stage completed!")
	print("# Doors detected:")
	for block in image_blocks:
		print("# DOOR",block[2])

	print("# 2nd stage ignition: Control Point Extraction  #")
	print("# Calculating...")
	save_image_parts(image_blocks,"cv")
	XYZ,xy = algorithm1(image_blocks)
	# testing_function(image_blocks)
	print("# The following Control Points were gathered:")
	print("#")
	print("# Space coordinates (mm):")
	print("# XYZ =",XYZ)
	print("#")
	print("# Picture coordinates (pixels):")
	print("# xy =",xy)
	print("#")
	print("# 3rd stage ignition: Position Estimation  #")
	print("# Calculating...")
	x0,y0,z0,res = position_estimation(XYZ,xy)
	print("# SUCCESS!!!!!")
	print("# Robot Position    is: X:",round(x0,3),"m , Y:",round(y0,3),"m , Z:",round(z0,3),"m")
	print("# Residual =",res,"m")
	print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
	print("#-------------------------------------------------#")


def check_crop():
	important_blocks = [(np.array([0.44808996, 0.13009809, 0.60111403, 0.22902723]), 0.99061674, 4)]
	PATH_TO_REF_IMAGES_DIR,PATH_TO_TEST_IMAGES_DIR = get_ref_test_image_path()
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
	image = cv.imread(TEST_IMAGE_PATHS[0],cv.COLOR_BGR2GRAY)
	image_blocks = seperate_blocks(image,important_blocks)


if __name__ == '__main__':
	main()
	# check_crop()