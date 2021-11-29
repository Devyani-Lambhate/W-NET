'''
This code marks the eddies that might are falsely marked in the ground truth images.
Our defination of a falsely marked eddy is as follows-

An eddy is said to be falsely marked if it is not present in the consecutive four images.
'''

import cv2
import numpy as np
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

s1=0
s2=0


path='merged/ALL_GT/'
#sst_fnames = sorted(os.listdir(os.path.join(path,'SST')))
#ssh_fnames = sorted(os.listdir(os.path.join(path, 'SSH')))

# Compare file names from GT folder to file names from SST and SSH folder Add it to the list image_arr if it exists in all the 3 folders (SST,SSH,GT)
image_arr=[]
for gt_fname in sorted(os.listdir(path)):
    #if gt_fname in sst_fnames and gt_fname in ssh_fnames:
	     image_arr.append(gt_fname)

print(len(image_arr))

r=2
i=4
for i in range(len(image_arr)-4):
	image = Image.open(path+image_arr[i])

	true_image=cv2.imread(path+image_arr[i],0)
	true_image = true_image.astype('uint8')

	# img_left_3 img_left_2 img_left_1 image img_right_1 img_right_2 img_right_3 

	img_left_1= cv2.imread(path+image_arr[i-1],0)
	img_left_1= img_left_1.astype('uint8')

	img_left_2= cv2.imread(path+image_arr[i-2],0)
	img_left_2= img_left_2.astype('uint8')

	img_left_3= cv2.imread(path+image_arr[i-3],0)
	img_left_3= img_left_3.astype('uint8')

	img_right_1= cv2.imread(path+image_arr[i+1],0)
	img_right_1= img_right_1.astype('uint8')

	img_right_2= cv2.imread(path+image_arr[i+2],0)
	img_right_2= img_right_1.astype('uint8')

	img_right_3= cv2.imread(path+image_arr[i+3],0)
	img_right_3= img_right_1.astype('uint8')




	
	'''
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(true_image, connectivity=4)
	sizes = stats[1:, -1]; nb_components = nb_components - 1
	img2 = np.zeros((output.shape),dtype = np.uint8)
	threshold=50
	#for every component in the image, you keep it only if it's above threshold
	for i in range(0, nb_components):
	    if sizes[i] >= threshold:
		    img2[output == i + 1] = 255
	'''
		    
	nb_components_true, output_true, stats_true, centroids_true = cv2.connectedComponentsWithStats(true_image, connectivity=4)

	nb_components_left_1, output_left_1, stats_left_1, centroids_left_1 = cv2.connectedComponentsWithStats(img_left_1, connectivity=4)
	nb_components_left_1, output_left_2, stats_left_2, centroids_left_2 = cv2.connectedComponentsWithStats(img_left_2, connectivity=4)
	nb_components_left_1, output_left_3, stats_left_3, centroids_left_3 = cv2.connectedComponentsWithStats(img_left_3, connectivity=4)
	nb_components_left_1, output_right_1, stats_right_1, centroids_right_1 = cv2.connectedComponentsWithStats(img_right_1, connectivity=4)
	nb_components_left_2, output_right_2, stats_right_2, centroids_right_2 = cv2.connectedComponentsWithStats(img_right_2, connectivity=4)
	nb_components_left_1, output_right_3, stats_right_3, centroids_right_3 = cv2.connectedComponentsWithStats(img_right_3, connectivity=4)

	'''

	4 possible cases

	CASE 1: * # # #   Look for the next three images
	CASE 2: # * # #   Look for the previous image and upcoming 2 images
	CASE 3: # # * #   Look for the previous 2 images and the upcoming 1 image
	CASE 4: # # # *   Look for the previous 3 images

	'''
	########### CASE 1 ##############

		
	case1=0
	case2=0
	case3=0
	case4=0

	sizes = stats_true[1:, -1]
	print(sizes,centroids_true)

	for c_t in centroids_true[2:]:

		#################CASE 1#################
		flag_hit_1=0

		for c_p in centroids_left_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_1=1
				break;

		flag_hit_2=0
		for c_p in centroids_left_2[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_2=1
				break;

		flag_hit_3=0
		for c_p in centroids_left_3[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_3=1
				break;

		if(flag_hit_3==1 and flag_hit_2==1 and flag_hit_1==1):
			case1=1

		############CASE 2#####################
		flag_hit_1=0

		for c_p in centroids_left_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_1=1
				break;

		flag_hit_2=0
		for c_p in centroids_left_2[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_2=1
				break;

		flag_hit_3=0
		for c_p in centroids_right_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_3=1
				break;

		if(flag_hit_3==1 and flag_hit_2==1 and flag_hit_1==1):
			case2=1


		############CASE 3#####################
		flag_hit_1=0

		for c_p in centroids_left_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_1=1
				break;

		flag_hit_2=0
		for c_p in centroids_right_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_2=1
				break;

		flag_hit_3=0
		for c_p in centroids_right_2[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_3=1
				break;

		if(flag_hit_3==1 and flag_hit_2==1 and flag_hit_1==1):
			case3=1


		############CASE 4#####################
		flag_hit_1=0

		for c_p in centroids_left_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_1=1
				break;

		flag_hit_2=0
		for c_p in centroids_right_1[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_2=1
				break;

		flag_hit_3=0
		for c_p in centroids_right_2[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_hit_3=1
				break;

		if(flag_hit_3==1 and flag_hit_2==1 and flag_hit_1==1):
			case4=1


		if(case1==1 or case2==1 or case3==1 or case4==1):
			continue
		else:

			
			x,y=c_t
			draw = ImageDraw.Draw(image)
			leftUpPoint = (x-r, y-r)
			rightDownPoint = (x+r, y+r)
			twoPointList = [leftUpPoint, rightDownPoint]
			draw.ellipse(twoPointList, fill=(255))
	image=np.array(image)

	image[image==1]=50
	image[image==2]=100
	image[image==3]=150

	im = Image.fromarray(image.astype('uint8'),mode="L")
	
	im.save('false_eddies/'+image_arr[i])




'''		
				
		if(flag_miss==0):
			avg_miss_size+=sizes[j]
			#print(sizes[i])
		j+=1

	if(hit!=0):
		avg_hit_size/=hit
	miss=nb_components_true-1-hit
	if(miss!=0):
		avg_miss_size/=miss


	s1+=avg_hit_size
	s2+=avg_miss_size
	#print('avg_miss_size',avg_miss_size)
	#print('avg_hit_size',avg_hit_size)
			
print('hit',s1/len(image_arr))
print('miss',s2/len(image_arr))

'''
   
