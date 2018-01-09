import model2
import preprocessQ2 as pp
import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def training(a):
	x_train,y_train,x_test,y_test=pp.preprocess()
	x_train=x_train.astype(np.float32)
	x_test=x_test.astype(np.float32)
	y_train=y_train.astype(np.float32)
	y_test=y_test.astype(np.float32)
	x_train=x_train/255
	x_test=x_test/255
	print(x_train.shape,y_train.shape)	
	a.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_test[:int(0.5*len(x_test))],y_test[:int(0.5*len(y_test))]))
	print(a.evaluate(x_test[int(0.5*len(x_test)):],y_test[int(0.5*len(y_test)):],batch_size=32))
	a.save_weights(os.getcwd()+'/4c5dweights.h5')
	'''for x in x_test:
		l=[x]
		l=np.asarray(l)	
		y=a.predict(l,batch_size=1)
		temp=x*255
		temp=temp.astype(np.uint8)
		cv2.rectangle(temp,(y[0],y[2]),(y[1],y[3]),(0,0,255),2)
		cv2.imshow('image',temp)
		cv2.waitKey(1000)
	'''		
	return a
#model = resnet34.modeling(50,50,3)
'''
dir_path = '/home/ankur/VOCdevkit/VOC2010/Annotations'
dir_path2 = '/home/ankur/VOCdevkit/VOC2010/JPEGImages'
data = parsing(dir_path,dir_path2)
'''
model1=model2.modeling(500,500,3)
training(model1)

