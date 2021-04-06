from matplotlib import pyplot as plt
import os
from PIL import Image
import os
import cv2
import numpy as np

# Our dir names
sst_dir = 'SST/'
ssh_dir = 'SSH/'
gt_dir = 'GT/' 
root=  'merged/test_data/'


path='merged/test_data'
sst_fnames = sorted(os.listdir(os.path.join(path,'SST')))
ssh_fnames = sorted(os.listdir(os.path.join(path, 'SSH')))

# Compare file names from GT folder to file names from RGB:
image_arr=[]
for gt_fname in sorted(os.listdir(os.path.join(path, 'GT'))):
    if gt_fname in sst_fnames and gt_fname in ssh_fnames:
	     image_arr.append(gt_fname)

print(len(image_arr))
'''
# Get all the filenames from RGB folder
sst_fnames = sorted(os.listdir(os.path.join(root, sst_dir)))
ssh_fnames = sorted(os.listdir(os.path.join(root, ssh_dir)))
'''
# Compare file names from GT folder to file names from RGB:

 
# Create an image with text on it
#img = np.zeros((100,400),dtype='uint8')
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'TheAILearner',(5,70), font, 2,(255),5,cv2.LINE_AA)

for image_name in image_arr:


	img = cv2.imread('true/'+image_name,0)
	print(image_name)
	print(img)
	#img = cv.cvtColor(img, cv.grey)
	img1 = img.copy()

	# Structuring Element
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	# Create an empty output image to hold values
	thin = np.zeros(img.shape,dtype='uint8')
	 
	# Loop until erosion leads to an empty set
	while (cv2.countNonZero(img1)!=0):
	    # Erosion
	    erode = cv2.erode(img1,kernel)
	    # Opening on eroded image
	    opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
	    # Subtract these two
	    subset = erode - opening
	    # Union of all previous sets
	    thin = cv2.bitwise_or(subset,thin)
	    # Set the eroded image for next iteration
	    img1 = erode.copy()
	    
	#cv2.imshow('original',img)
	#cv2.imshow('thinned',thin)
	thin[thin<150]=0
	im = Image.fromarray(thin)
	im = im.convert('RGB')
	im.save('Gulf Stream/True/'+image_name)
#plt.imshow(thin)
#plt.savefig('thin_true'+'20150105.png',dpi=300)
#cv2.waitKey(0)

'''
for gt_fname in sorted(os.listdir(os.path.join(root, gt_dir))):
	#if gt_fname in sst_fnames and gt_fname in ssh_fnames:
		print(gt_fname)
		image1 = cv.imread(root+gt_dir+gt_fname)

		#image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
		print(image1)
		image1[image1==1]=50
		image1[image1==2]=100
		image1[image1==3]=150
		plt.imshow(image1)
		plt.savefig('gt_images'+gt_fname,dpi=300) # To save figure
		plt.show()
'''
