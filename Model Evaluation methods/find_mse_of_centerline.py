'''
Finds the mean square error between the predicted and the ground truth gulf stream
'''

from __future__ import print_function
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
import tensorflow as tf
import math
from matplotlib import pyplot as plt
#sess = tf.InteractiveSession()

def load_data (is_real=False):

			if(is_real==True):
				folder = "Gulf Stream/True/"
			else:
				folder = "Gulf Stream/Pred/"

			onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

			train_files = []
			y_train = []
			for _file in onlyfiles:
			    train_files.append(_file)
			    label_in_file = _file.find("_")
			    
			print("Files in train_files: %d" % len(train_files))
			train_files=sorted(train_files)
			print("train_files")
			# Original Dimensions
			image_width = 512
			image_height = 512
			channels = 3

			dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
					     dtype=np.float32)

			i = 0
			size=512
			for _file in train_files:
			    
			    img = load_img(folder + "/" + _file)  # this is a PIL image
			    newsize = (size,size) 
			    # Convert to Numpy Array
			    x = img_to_array(img)  
			    x = x.reshape((size,size,3))
			    dataset[i] = x
			    i += 1
			    if i % 250 == 0:
			    	print("%d images to array" % i)
			print("All images to array!")
			print (dataset.shape)
			return dataset

def calculateDistance(i1, i2):
	i1[i1==150]=1
	i2[i2==150]=1
	return math.sqrt(np.sum((i1-i2)**2))
    


X=load_data(is_real=True)
print(X)
Y=load_data()
print(X.shape)
print(Y.shape)
mse=0
arr_mse=[]
arr_ssim=[]
for i in range(67):
	mse+=calculateDistance(X[i],Y[i])
	arr_mse.append(calculateDistance(X[i],Y[i]))
mse/=67

print('average_mean_square= ',mse)
