import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

sys.path.append('..')
from detection.door_detection import find_test_doors
from compvis_utils.control_point_extraction import sift

def get_ref_test_image_path():
	PROJECT_HOME_FOLDER = 	"../../"
	PATH_TO_REF_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/reference"
	REF_IMAGE_PATHS = [os.path.join(PATH_TO_REF_IMAGES_DIR,im) for im in os.listdir(PATH_TO_REF_IMAGES_DIR)]
	print(REF_IMAGE_PATHS)	
	PATH_TO_TEST_IMAGES_DIR = PROJECT_HOME_FOLDER + "images/test"
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

	print(TEST_IMAGE_PATHS)
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
	for i in range(num_detected):
		if blocks[classes[i]-1] == ():
			blocks[classes[i]-1] = (boxes[i],scores[i],classes[i])
		elif scores[i] > blocks[classes[i]-1][1]:
			blocks[classes[i]-1] = (boxes[i],scores[i],classes[i])
	for block in blocks:
		print(block)
		if block[1] > 0.4:
			important_blocks.append(block)

	return important_blocks

def chop_block(or_image,block_bounds):
	#print(or_image)

	#print(or_image.size)
	image_pil = Image.fromarray(or_image)
	im_width,im_height = image_pil.size
	# print(im_width,im_height)
	# image_pil.show()
	ymin = block_bounds[0]
	xmin = block_bounds[1]
	ymax = block_bounds[2]	
	xmax = block_bounds[3]	
	left, bottom, right, top = xmin * im_width, ymax * im_height, xmax * im_width, ymin * im_height
	# print(left, top, right, bottom)
	chopped_pil = image_pil.crop((int(left), int(top), int(right), int(bottom)))
	# chop_im = load_image_into_numpy_array(chopped_pic)
	#chopped_pil.show()
	#chopped_pil.save('/home/smarn/thesis/robot_localisation/images/detected_doors/chopped.jpg')
	cropped_im = load_image_into_numpy_array(chopped_pil)
	return cropped_im


def save_image_parts(image_blocks):

	for block in image_blocks:
		tmp_pil = Image.fromarray(block[0])
		tmp_label = block[2]
		image_name = 'DOOR' + str(tmp_label) + '.jpg'
		tmp_pil.save('../../images/detected_doors/'+image_name)



def seperate_blocks(or_image,imp_bl):
	# Returns all detected doors in tuple of form:
	# image_block = (door-image-part in np-array,score,label)
	image_blocks = []
	for block in imp_bl:
		# print('DOOR')
		print(block[2])
		print(block[1])
		image_blocks.append((chop_block(or_image,block[0]),block[1],block[2]))
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
	# PATH_TO_TEST_IMAGES_DIR = '../../images/test'
	PATH_TO_REF_IMAGES_DIR,PATH_TO_TEST_IMAGES_DIR = get_ref_test_image_path()
	image,boxes,scores,classes,num_detections = find_test_doors(PATH_TO_TEST_IMAGES_DIR)
	
	important_blocks = exclude_missdetections(boxes,scores,classes,num_detections)
	# print(important_blocks)
	image_blocks = seperate_blocks(image,important_blocks)
	print("image block len:")
	print(len(image_blocks))
	save_image_parts(image_blocks)
	#test_ref_pairs = match_with_reference(image_blocks,PATH_TO_REF_IMAGES_DIR)
	print("pairs len:")
	#print(len(test_ref_pairs))
	#print(test_ref_pairs)
	# plt.imsave('/home/smarn/thesis/images/detected_doors/image.jpg',image)
	# sift(image_blocks[0][1])



if __name__ == '__main__':
	main()