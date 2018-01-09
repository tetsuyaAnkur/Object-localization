import xml.etree.ElementTree
import os

def parsing(a,b):

	l = {}
	length = 0
	for file in os.listdir(a):
		key = file[0:len(file)-4] + ".jpg"
		root = xml.etree.ElementTree.parse(a+"/"+file).getroot()
		
		count = 0
		
		for x in root.findall('object'):
			count += 1
		if count==1:	
			length += 1
			#print("file = " + file)
			m = []
			for y in root.find('object'):
				for y in x.findall('bndbox'):
					n = []
					xmin = int(y.find('xmin').text)			
					ymin = int(y.find('ymin').text)
					xmax = int(y.find('xmax').text)
					ymax = int(y.find('ymax').text)
					n.append(xmin)
					n.append(xmax)
					n.append(ymin)
					n.append(ymax)
					m.append(n)
			
			l[key] = m[0]
	#print(length)
	return(l)

#print(l)
#print(len(l.keys()))

#dir_path = '/home/ankur/VOCdevkit/VOC2010/Annotations'
#dir_path2 = '/home/ankur/VOCdevkit/VOC2010/JPEGImages'

#data = parsing(dir_path,dir_path2)

#print(data)
