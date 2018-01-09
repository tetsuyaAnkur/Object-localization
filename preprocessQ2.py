def preprocess():
	import os
	import cv2
	import numpy as np
	import time
	import random
	data={}
	for filename in os.listdir(os.getcwd()+'/JPEGImages'):
		data[filename]=cv2.imread(os.getcwd()+'/JPEGImages/'+filename)
		#print(type(data[filename]))
	#print(len(data.keys()))
	max_h=0
	max_w=0
	for key in data.keys():
		#print(type(data[key]))
		if(data[key].shape[0]>max_h):
			max_h=data[key].shape[0]
		if(data[key].shape[1]>max_w):
			max_w=data[key].shape[1]
	#print(max_h,max_w)
	datanew={}
	for key in data.keys():
		#print(data[key].shape)
		#print(data[key].dtype)	
		datanew[key]=np.zeros((max_h,max_w,3),dtype=np.uint8)
		#print(datanew[key].dtype)	
		datanew[key][:data[key].shape[0],:data[key].shape[1],:]=data[key]
	del data
	#alldata=random.sample(datanew.keys(),3000)
	import parseQ2 as px
	l=px.parsing(os.getcwd()+'/Annotations',os.getcwd()+'/JPEGImages')
	classes={}
	#for x in l.keys():
	#	print(x)
	#for x in datanew.keys():
	#	print x
	datanew1={}
	for x in l.keys():
		for y in datanew.keys():
			if((y) == x):
				datanew1[y]=datanew[y]
	datanew=datanew1
	for x in datanew.keys():
		if l[x][0]  not in classes.keys():
			classes[l[x][0]]=1
		#print(l[x][0])
	setting={}
	i=0
	for x in classes.keys():
		setting[x]=i
		i=i+1
	#print(setting)
	labels={}
	#print(len(datanew.keys()))
	for x in datanew.keys():
		#print(l[x])
		temp=np.zeros(4)
		temp[:]=l[x]
		labels[x]=temp
		#print(temp)
	alldata=random.sample(datanew.keys(),3000)
	train_data=[]
	train_labels=[]
	for x in alldata[:int(0.9*3000)]:
		train_data.append(datanew[x])
		train_labels.append(labels[x])
	test_data=[]
	test_labels=[]
	for x in alldata[int(0.9*3000):]:
		test_data.append(datanew[x])
		test_labels.append(labels[x])
	del datanew
	del labels
	del datanew1
	del alldata
	#print(setting)
	return (np.asarray(train_data),np.asarray(train_labels),np.asarray(test_data),np.asarray(test_labels))

