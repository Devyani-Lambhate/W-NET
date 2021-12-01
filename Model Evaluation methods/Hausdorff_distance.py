'''
Finds the Haussdorff distance, mean curve distance and median curve distance between the predicted and the ground truth gulfstream
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import seaborn as sns


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

#image_arr=['20150105.png','20160708.png','20171020.png','20180430.png']
print(image_arr)


def main():

    # 1. Import pictures
    min_list=[]
    max_list=[]
    mean_list=[]
    for image in image_arr:
    
	    img_cs1 = cv2.imread("Gulf Stream/True/"+image,0)
	    img_cs2 = cv2.imread("Gulf Stream/Pred/"+image,0)
	    #print(np.unique(img_cs1))
	    #print(img_cs2.shape)


	    true=[]
	    pred=[]

	    for i in range(0,410):
	    	for j in range(512):
	    	    if(img_cs1[i,j]==150):
	    	    	true.append([i,j])
	    	    if(img_cs2[i,j]==150):
	    	    	pred.append([i,j])

	    min_arr=[]
	    

	    for pix_pos_t in true:
	    	x_t,y_t=pix_pos_t
	    	min=10000
	    	for pix_pos_p in pred:
	    		x_p,y_p=pix_pos_p
	    		d=math.sqrt((x_p-x_t)**2+(y_p-y_t)**2)
	    		if(d<min):
	    			min=d
	    	min_arr.append(min)
	    	
	    max_list.append(np.max(min_arr))
	    min_list.append(np.min(min_arr))
	    mean_list.append(np.mean(min_arr))

	    ax=sns.distplot(min_arr)
	    fig=ax.get_figure()
	    #fig.savefig('ksdensity'+image)



   
    print(np.mean(max_list),np.mean(min_list),np.mean(mean_list),np.median(mean_list))
    print(mean_list)





    #cnt_hand = get_contours(img_hand)


    
    

    
if __name__ == '__main__':
    main()
