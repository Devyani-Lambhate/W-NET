# import image utilities
'''
This code performs data augmentation on the SST , SSH and GT images
by every 45 degree. It then saves all the rotated images by 
appending the rotation angle with the image name
'''

from PIL import Image
from scipy import ndimage, misc

# import os utilities
import os

# define a function that rotates images in the current directory
# given the rotation in degrees as a parameter
def rotateImages(rotationAmt):
  """Rotates and saves all the images with the given rotation amount

    Parameters
    ----------
    rotationAmt : int
        Degree by which we need to rotate
    
    """
  # for each image in the current directory
  path_gt="merged/2019_20/GT/"
  #path_ssh="merged/winter/train_data/SSH/"
  path_ssh="merged/2019_20/SST/"
  
  #path_save_sst="merged/train_data_augmented/1km_SST/"
  #path_save_ssh="merged/winter/train_data_augmented/SSH/"
  path_save_ssh="merged/2019_20_augmented/SST/"
  
  
  
  #for image in os.listdir(path_gt):
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

#Iterate over all the rotation angles
for i in range(len(rot)):
	rotateImages(rot[i])

