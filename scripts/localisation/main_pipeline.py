import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

sys.path.append('..')
from detection.door_detection import find_test_doors

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def exclude_missdetections(boxes,scores,classes,num_detections):
	important_blocks = []
	for i in range(len(classes)):
		if scores[i] > 0.50:
			important_blocks.append((boxes[i],scores[i],classes[i]))

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
	chopped_pil.show()
	# chopped_pil.save('/home/smarn/thesis/images/detected_doors/chopped.jpg')
	#chopped_pic.show()
	return chopped_pil

def seperate_blocks(or_image,imp_bl):
	image_blocks = []
	labels = []
	for block in imp_bl:
		print(block[0])
		image_blocks.append((chop_block(or_image,block[0]),block[1],block[2]))
	return image_blocks

def main():

	# Bring test image and image blocks
	PATH_TO_TEST_IMAGES_DIR = '../../images/test'
	image,boxes,scores,classes,num_detections = find_test_doors(PATH_TO_TEST_IMAGES_DIR)
	
	important_blocks = exclude_missdetections(boxes,scores,classes,num_detections)
	#print(important_blocks)
	image_blocks = seperate_blocks(image,important_blocks)

	print(image_blocks)
	#plt.imsave('/home/smarn/thesis/images/detected_doors/image.jpg',image)



if __name__ == '__main__':
	main()
