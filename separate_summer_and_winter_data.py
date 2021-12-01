
# This code separates the winter month images and Summer month images. And saves them in a different folder
from PIL import Image

# import os utilities
import os

# define a function that rotates images in the current directory
# given the rotation in degrees as a parameter
def extract(season):
  # for each image in the current directory
  path_sst="merged/val_data/SST/"
  path_ssh="merged/val_data/SSH/"
  path_gt="merged/val_data/GT/"
  
  path_save_sst="merged/winter/val_data/SST/"
  path_save_ssh="merged/winter/val_data/SSH/"
  path_save_gt="merged/winter/val_data/GT/"
  
  for image in os.listdir(path_gt):
    # open the image
    if(season=='summer'):
    	if(int(image[4:6]) in (5,6,7,8,9)):
	    	img = Image.open(path_ssh+image)
	    	img.save(path_save_ssh+image[0:8]+'.png')
	    	img.close()
	    	print(image)
    elif(season=='winter'):
    	if(int(image[4:6]) in (10,11,12,1,2,3,4)):
    		img = Image.open(path_gt+image)
    		img.save(path_save_gt+image[0:8]+'.png')
    		img.close()
    		print(image)
    
    
    # rotate and save the image with the same filename
    #img=img.rotate(rotationAmt,resample=Image.BILINEAR)
    #save
    
    # close the image
    
    
# examples of use
season='winter'
extract(season)
