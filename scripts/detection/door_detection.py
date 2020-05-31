#!/usr/bin/env python
# coding: utf-8

import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import cv2

import tkinter

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
sys.path.append('/home/smarn/tensorflow/models/research/object_detection')

from object_detection.utils import ops as utils_ops
 
from object_detection.utils import label_map_util
 
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def find_test_doors(PATH_TO_TEST_IMAGES_DIR):

	# Configuring Model paths
	# TENSORFLOW_FOLDER = sys.path[-1]
	PROJECT_FOLDER = '../../'

	MODEL_FOLDER = 'tensorflow-object_detection/MODEL/'
	MODEL_NAME = 'lab_doors3_graph_faster_RCNN_resnet'
	
	MODEL_PATH = PROJECT_FOLDER + MODEL_FOLDER + MODEL_NAME
	
	# Configuring path to NN's inference graph
	PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
	 
	# Configuring path to data labels

	PATH_TO_LABELS = PROJECT_FOLDER + 'tensorflow-object_detection/' + 'lab_doors-detection.pbtxt'
	# PATH_TO_LABELS = os.path.join(TENSORFLOW_FOLDER,'data', 'lab_doors-detection.pbtxt')


	NUM_CLASSES = 9
	
	# Initializing detection
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	    od_graph_def = tf.GraphDef()
	    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	        serialized_graph = fid.read()
	        od_graph_def.ParseFromString(serialized_graph)
	        tf.import_graph_def(od_graph_def, name='')


	#load the labels
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)



	#PATH_TO_TEST_IMAGES_DIR = 'home/test_images/test3'
	# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2) ]
	TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,im) for im in os.listdir(PATH_TO_TEST_IMAGES_DIR) ]
	IMAGE_SIZE = (12, 8)
	#TEST_IMAGE_PATHS

	i = 1
	with detection_graph.as_default():
	    with tf.Session(graph=detection_graph) as sess:
	        for image_path in TEST_IMAGE_PATHS:
	            image = Image.open(image_path)
	            #print(image)
	            image_np = load_image_into_numpy_array(image)
	            original_image = load_image_into_numpy_array(image)
	            #print(image_np)
	            #plt.show(image_np)
	            # original_image = image_np[:]
	            image_np_expanded = np.expand_dims(image_np, axis=0)
	            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	            
	            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	            scores = detection_graph.get_tensor_by_name('detection_scores:0')
	            
	            classes = detection_graph.get_tensor_by_name('detection_classes:0')
	            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	            
	            (boxes,scores,classes,num_detections) = sess.run(
	                [boxes,scores,classes,num_detections],
	                feed_dict={image_tensor: image_np_expanded})
	            #print(np.squeeze(scores))
	            
	            vis_util.visualize_boxes_and_labels_on_image_array(
	                image_np,
	                np.squeeze(boxes),
	                np.squeeze(classes).astype(np.int32),
	                np.squeeze(scores),
	                category_index,
	                min_score_thresh = 0.40,
	                use_normalized_coordinates=True,
	                line_thickness=8)
	            #plt.figure(figsize=IMAGE_SIZE)
	            #plt.imsave('/home/smarn/thesis/images/detected_doors/image{}.jpg'.format(i),image_np)
	            plt.imsave('/home/smarn/thesis/robot_localisation/images/report/image.jpg',image_np)
	            i = i + 1
	            #plt.show()

	return original_image,np.squeeze(boxes),np.squeeze(scores),np.squeeze(classes).astype(np.int32),np.squeeze(num_detections)

if __name__ == '__main__':
	#results = find_test_doors()
	print("Succes")