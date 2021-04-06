import cv2
import numpy as np
import os
import math

s1=0
s2=0


path='merged/test_data'
sst_fnames = sorted(os.listdir(os.path.join(path,'SST')))
ssh_fnames = sorted(os.listdir(os.path.join(path, 'SSH')))

# Compare file names from GT folder to file names from RGB:
image_arr=[]
for gt_fname in sorted(os.listdir(os.path.join(path, 'GT'))):
    if gt_fname in sst_fnames and gt_fname in ssh_fnames:
	     image_arr.append(gt_fname)

print(len(image_arr))



def find_centroid_difference(ti,pi):
	nb_components_true, output_true, stats_true, centroids_true = cv2.connectedComponentsWithStats(ti, connectivity=4)
	sizes = stats_true[1:, -1]; nb_components_true = nb_components_true - 1
	img2 = np.zeros((output_true.shape),dtype = np.uint8)
	threshold=5000
	#for every component in the image, you keep it only if it's below threshold
	for i in range(0, nb_components_true):
	    if sizes[i] <= threshold:
		    img2[output_true == i + 1] = 255

	nb_components_true, output_true, stats_true, centroids_true = cv2.connectedComponentsWithStats(img2, connectivity=4)

	nb_components_pred, output_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(pi, connectivity=4)

	for c_t in centroids_true:
		flag_miss=0
		min_dist=10000
		for c_p in centroids_pred:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<min_dist):
				min_dist=dist

	return min_dist


def count_eddies(l,e):

	true_image=cv2.imread('test/'+e+'/'+l,0)
	true_image = true_image.astype('uint8')



	pred_image=cv2.imread('pred/'+e+'/'+l,0)
	pred_image = pred_image.astype('uint8')
	pred_image[pred_image==50]=150
	pred_image[pred_image==100]=150

	#print(np.unique(pred_image),"!!!!!")

	p_image=np.add(true_image,pred_image)

	eddies_list,pix=np.unique(true_image,return_counts=True)
	p_eddies_list,p_pix=np.unique(p_image,return_counts=True)
	#print(p_eddies_list)
	
	count_correctly_pred_eddies=0
	count_correctly_pred_eddies_with_diameter_75km_or_more=0
	count_eddies_75_km=0
	count_eddies=len(eddies_list)-1

	avg_hit_size=0
	avg_miss_size=0

	avg_centroid_difference=0
	avg_area_difference=0
	avg_diameter_difference=0


	for i in range(1,len(eddies_list)):
		
		
		x=np.count_nonzero(p_image == 150+eddies_list[i])

		if((math.sqrt(pix[i]/math.pi))>6.81):
					count_eddies_75_km+=1

		if(x/pix[i]>0):
				count_correctly_pred_eddies+=1
				avg_hit_size+=pix[i]
				y=eddies_list[i]
				ti=np.copy(true_image)
				pi=np.copy(pred_image)
				ti[ti!=y]=0
				#print(np.unique(ti))
				avg_centroid_difference+=find_centroid_difference(ti,pi)
				avg_area_difference+=abs(pix[i]-x)
				d1=2*math.sqrt(pix[i]/math.pi)
				d2=2*math.sqrt(x/math.pi)
				avg_diameter_difference+=(abs(d1-d2))/d1
				#print(abs(d1-d2))
				if(math.sqrt(pix[i]/math.pi)>6.81):
					count_correctly_pred_eddies_with_diameter_75km_or_more+=1


		else:
			avg_miss_size+=pix[i]

	if(count_correctly_pred_eddies==0):
		avg_hit_size=0
		avg_centroid_difference=0
		avg_area_difference=0
	else:
		avg_hit_size/=count_correctly_pred_eddies
		avg_centroid_difference/=count_correctly_pred_eddies
		avg_area_difference/=count_correctly_pred_eddies
		avg_diameter_difference/=count_correctly_pred_eddies

	if(len(eddies_list)-1-count_correctly_pred_eddies ==0):
		avg_miss_size=0
	else:
		avg_miss_size/=(len(eddies_list)-1-count_correctly_pred_eddies)

	#print('count_correctly_pred_eddies: ',count_correctly_pred_eddies)
	#print('count_eddies: ',count_eddies)

	#print(avg_diameter_difference)
	return count_eddies,count_eddies_75_km,count_correctly_pred_eddies,avg_hit_size,avg_miss_size,avg_centroid_difference,avg_area_difference,avg_diameter_difference,count_correctly_pred_eddies_with_diameter_75km_or_more


	
	'''
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(true_image, connectivity=4)
	sizes = stats[1:, -1]; nb_components = nb_components - 1
	img2 = np.zeros((output.shape),dtype = np.uint8)
	threshold=50
	#for every component in the image, you keep it only if it's above threshold
	for i in range(0, nb_components):
	    if sizes[i] >= threshold:
		    img2[output == i + 1] = 255
	
		    
	nb_components_true, output_true, stats_true, centroids_true = cv2.connectedComponentsWithStats(true_image, connectivity=4)

	pred_image=cv2.imread('pred/'+l,0)
	pred_image = pred_image.astype('uint8')

	nb_components_pred, output_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(pred_image, connectivity=4)

	hit=0
	j=0	
	avg_hit_size=0
	avg_miss_size=0
	
	sizes = stats_true[1:, -1]
	print(sizes,centroids_true)

	for c_t in centroids_true[2:]:
		flag_miss=0
		for c_p in centroids_pred[2:]:
			#print(c_t,c_p)
			dist = np.linalg.norm(c_t-c_p)
			if(dist<10):
				flag_miss=1
				avg_hit_size+=sizes[j]
				hit+=1
				break;
			
				
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
avg_count_eddies_75km=[]
avg_count_eddies=[]
avg_avg_diameter_difference=[]
for i in range(len(image_arr)):
	print(image_arr[i])
	count_edd,count_eddies_75_km,count_correctly_pred_eddies,_,_,_,_,avg_diameter_difference,count_correctly_pred_eddies_with_diameter_75km_or_more=count_eddies(image_arr[i],'warm')
	avg_avg_diameter_difference.append(avg_diameter_difference)
	#print(avg_diameter_difference)
	avg_count_eddies.append(count_correctly_pred_eddies/count_edd)
	avg_count_eddies_75km.append(count_correctly_pred_eddies_with_diameter_75km_or_more/count_eddies_75_km)



print(np.mean(avg_avg_diameter_difference))
#print('eddies count', np.mean(avg_count_eddies))
print('eddie count 75 km', np.mean(avg_count_eddies_75km))