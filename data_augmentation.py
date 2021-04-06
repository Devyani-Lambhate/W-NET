# import image utilities
from PIL import Image
from scipy import ndimage, misc

# import os utilities
import os

# define a function that rotates images in the current directory
# given the rotation in degrees as a parameter
def rotateImages(rotationAmt):
  # for each image in the current directory
  #path_sst="merged/train_data/1km_SST/"
  #path_ssh="merged/winter/train_data/SSH/"
  path_ssh="merged/train_data/low_res_SSH_recent/"
  
  #path_save_sst="merged/train_data_augmented/1km_SST/"
  #path_save_ssh="merged/winter/train_data_augmented/SSH/"
  path_save_ssh="merged/train_data_augmented/low_res_SSH_recent/"
  
  for image in os.listdir(path_ssh):
    # open the image
    print(image)
    img = Image.open(path_ssh+image)
    # rotate and save the image with the same filename
    img=img.rotate(rotationAmt,resample=Image.BILINEAR)
    #save
    img.save(path_save_ssh+image[0:8]+'_'+str(rotationAmt)+'.png')
    # close the image
    img.close()
    
# examples of use
rot=[  0. , 45. , 90., 135., 180., 225. ,270. ,315.]
for i in range(len(rot)):
	rotateImages(rot[i])

