'''
This code feathes all the SST maps, applies 
canny edge detection algorithm and saves the
resultant images
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

sst_dir = 'SST/'
ssh_dir = 'SSH/'
gt_dir = 'GT/'  
root='merged/val_data/'
sst_fnames = sorted(os.listdir(os.path.join(root, sst_dir)))
ssh_fnames = sorted(os.listdir(os.path.join(root, ssh_dir)))

# Compare file names from GT folder to file names from RGB:
for gt_fname in sorted(os.listdir(os.path.join(root, gt_dir))):

	if gt_fname in sst_fnames and gt_fname in ssh_fnames:
       
		print(root+sst_dir+gt_fname)
		img = cv2.imread(root+sst_dir+gt_fname,0)
		print(img)
		edges = cv2.Canny(img,80,85)

		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

		plt.savefig('canny/'+gt_fname)

