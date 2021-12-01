'''
Finds the mean and median path length difference between the predicted and the ground truth gulf stream
'''
import sys 
import cv2
import math
import numpy as np
import os
from skimage.morphology import area_opening
  
# Driver program
#Our dir names
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
#print(image_arr)


#Array that will store the mean path difference between all the examples
mean_path_difference=[]


#x_data=[]
#y_data=[]
#i=0
#starting_point=[0,0]
#start_c=0
def find_index(x,y,cordinates):
    """ Finds the index of cordinates[x,y] in the visited list

    Parameters
    ----------
    x: int
        x cordinate of the pixel
    y: int
        y cordinate of the pixel
    img_cordinates : list of list
        List of all cordinates where Gulfstream is present
        
        
    Returns
    -------
    i: int
    	index of cordinate
    
    """
	for i in range(len(cordinates)):
		if([x,y] == cordinates[i]):
			return i
			
			
def min_distance(pixel,img_cordinates,visited):
    """ Finds the next point to visit and update the visited list

    Parameters
    ----------
    pixel : list of int
        Cordinates of the current point
    img_cordinates : list of list
        List of all cordinates where Gulfstream is present
    visited: list of int
    	A list that keep a track of all visited cordinates
        
        
    Returns
    -------
    corx: int
        x cordinate of the next point to visit
    cory: int
        y cordinate of the next point to visit
    visited: list of int
        updated list of visited pixels
    
    """
		min_dis=500
		cor_x=0
		cor_y=0
		index_vis=0
		
		# From the list of all image pixels, find the pixel that is at the minimum distance
		for i in range(len(img_cordinates)):
			t=img_cordinates[i]

			if(visited[i]==0 or i==len(visited)-1):
				x_t1=t[0]
				y_t1=t[1]
				x_t2=pixel[0]
				y_t2=pixel[1]
				d=math.sqrt((x_t1-x_t2)**2+(y_t1-y_t2)**2)

				if(d<min_dis):
					min_dis=d
					cor_x=x_t1
					cor_y=y_t1
					index_vis=i
					
		# Mark this pixel as visited
		visited[index_vis]=1
		
		for cor in img_cordinates :
			if(cor[1] <cor_y ):
				ind=find_index(cor[0],cor[1],img_cordinates) 
				visited[ind]=1
				
		# Mark the surrounding 8 pixels as visited
		if([cor_x+1,cor_y+1] in img_cordinates):
			ind=find_index(cor_x+1,cor_y+1,img_cordinates) 
			visited[ind]=1
			
		if([cor_x-1,cor_y+1] in img_cordinates):
			ind=find_index(cor_x-1,cor_y+1,img_cordinates) 
			visited[ind]=1
			
		if([cor_x+1,cor_y-1] in img_cordinates):
			ind=find_index(cor_x+1,cor_y-1,img_cordinates) 
			visited[ind]=1
			
		if([cor_x-1,cor_y-1] in img_cordinates):
			ind=find_index(cor_x-1,cor_y-1,img_cordinates) 
			visited[ind]=1
		
		if([cor_x,cor_y+1] in img_cordinates):
			ind=find_index(cor_x,cor_y+1,img_cordinates) 
			visited[ind]=1
			
		if([cor_x,cor_y-1] in img_cordinates):
			ind=find_index(cor_x,cor_y-1,img_cordinates) 
			visited[ind]=1
			
		if([cor_x+1,cor_y] in img_cordinates):
			ind=find_index(cor_x+1,cor_y,img_cordinates) 
			visited[ind]=1
			
		if([cor_x-1,cor_y] in img_cordinates):
			ind=find_index(cor_x-1,cor_y,img_cordinates) 
			visited[ind]=1
			
			
		return cor_x,cor_y,visited

def find_path_length(start_p,image,img_cordinates,name):

    """ Finds path length of centre line from the given image and returns it

    Parameters
    ----------
    start_p : list of int
        Cordinates of starting point
    image : list of list
        Image containing Gulfstream centerline
    img_cordinates : list of list
        List of all cordinates where Gulfstream is present
    name: str
    	name of the image
        
        
    Returns
    -------
    path_length
        an integer representing path length
    """


	
	start_point=img_cordinates[start_p]
	end_point=img_cordinates[-1]
	path_length=0
	pixel=start_point

	#print('start_point',start_point)
	
	#Marks the visited points
	visited=[0 for i in range(len(img_cordinates))]

	#print('end_point',end_point)
				
			
	visited[0]=1
	plot_image=[[0]*512]*512
	plot_image=np.array(plot_image)
	#plot_image[1,2]=140
	plot_image[start_point[0]][start_point[1]]=130
	m=0
	while pixel!=end_point:
		#print(pixel)
		m+=2
		

		#Find the next nearest point using min_distance
		next_point_x,next_point_y,visited=min_distance(pixel,img_cordinates,visited)
		#print(next_point_x,next_point_y)
		
		#Calculate the distance between previous visited point and the current point
		dis=math.sqrt((pixel[0]-next_point_x)**2+(pixel[1]-next_point_y)**2)
		# Add the distance to the pathlength
		path_length+=dis
		pixel=[next_point_x,next_point_y]
		plot_image[pixel[0]][pixel[1]]=100+m
		
	#print(plot_image)
	
	cv2.imwrite('plot_image'+name+'.png',plot_image)
	
	return path_length
		

# This loop iterates over all the examples
for image in image_arr:
  #if(i==0):
    #i+=1
    img=[]
    res=[]

    #img.append(cv2.imread("Gulf Stream/True/"+image,0))
    #img.append(cv2.imread("Gulf Stream/Pred/"+image,0))
   



    #Load the True Gulfstream
    img_t = cv2.imread("Gulf Stream/True/"+image,0)
    img_true=area_opening(img_t, area_threshold=3, connectivity=1, parent=None, tree_traverser=None)

     
    #Load the predicted Gulfstream 
    img_p = cv2.imread("Gulf Stream/Pred/"+image,0)
    img_pred=area_opening(img_p, area_threshold=3, connectivity=1, parent=None, tree_traverser=None)

 
       
    #print(np.unique(img_cs1))
    #print(img_cs2.shape)


    true=[]
    pred=[]

    start_point_t=0
    start_point_p=0

    #Starting point is added
    true.append([410,88])
    pred.append([410,88])
    
    for i in range(0,512):
        for j in range(0,410):
            if(img_true[j,i]!=0): #All the pixels where true Gulfstream centerline is present is appended in the true list
                true.append([j,i])
                #if(i<=start_point_t):
                	#start_point_t=len(true)-1
    
    for i in range(0,512):
        for j in range(0,410):
            if(img_pred[j,i]!=0):
                pred.append([j,i]) #All the pixels where predicted Gulfstream centerline is present is appended in the pred list
                #if(i<=start_point_p):
                	#start_point_p=len(pred)-1
    print(true[0],pred[0])
                    
    path_length_true=find_path_length(0,img_true,true,image+'1')         
    path_length_pred=find_path_length(0,img_pred,pred,image+'2')         

	
    print(abs(path_length_true-path_length_pred))
    c=0
    if(abs(path_length_true-path_length_pred) <200):
    	mean_path_difference.append(abs(path_length_true-path_length_pred)/path_length_true)
    else:
    	c+=1


print('mean path difference',np.median(mean_path_difference),np.mean(mean_path_difference),c)
#print(mean_path_difference)
#print(np.array(x_data).shape)
#x=np.linspace(0,512, 512)
