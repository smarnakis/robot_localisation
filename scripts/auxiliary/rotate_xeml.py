#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET

directory = '.'
for filename in os.listdir(directory):
	if filename.endswith(".xml"):
		print(filename)
		name = filename
		if ('rot' in filename) or ('ups' in filename): 
			tree = ET.parse(filename)
			root = tree.getroot()
			for child in root:
				if child.tag == 'filename':
					filelist = filename.split('.')
					filename = filelist[0]+".jpg"
					child.text = filename
				if child.tag == 'size':
					if 'rot' in filename:
						tmp = child[0].text
						child[0].text = child[1].text
						child[1].text = tmp
					width = child[0].text
					height = child[1].text
				if child.tag == 'path':
					#print(child.text)
					path = child.text.split('/')
					path[-1] = filename
					child.text = '/'.join([str(elem) for elem in path])
					#print(child.text) 

			for xml_object in root.findall('object'):
					box = xml_object.find('bndbox')
					cur_y_min = box.find('ymin').text
					cur_x_max = box.find('xmax').text
					cur_x_min = box.find('xmin').text
					cur_y_max = box.find('ymax').text
					if filename[:4] == 'rot1':
						#print('rot1\n')
						box.find('ymin').text = cur_x_min
						box.find('ymax').text = cur_x_max
						box.find('xmin').text = str(int(width) - int(cur_y_max))
						box.find('xmax').text = str(int(width) - int(cur_y_min))
					elif filename[:4] == 'ups_':
						#print('ups\n')
						box.find('ymax').text = str(int(height) - int(cur_y_min))
						box.find('ymin').text = str(int(height) - int(cur_y_max))
						box.find('xmax').text = str(int(width) - int(cur_x_min))
						box.find('xmin').text = str(int(width) - int(cur_x_max))
					elif filename[:4] == 'rot2':
						#print('rot2\n')
						box.find('ymin').text = str(int(height) - int(cur_x_max))
						box.find('ymax').text = str(int(height) - int(cur_x_min))
						box.find('xmin').text = cur_y_min
						box.find('xmax').text = cur_y_max
						

					else:
						print('do_nothing') 

			tree.write(name)
		#print(cur_y_max,cur_x_max,cur_y_min,cur_y_max)

