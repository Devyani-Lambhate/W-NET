
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:48:57 2020

@author: raghav
"""

'''
to convert nc files to RGB images
'''
nn=200

from netCDF4 import Dataset
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
from NC_to_images_ssh import NC_image_ssh
def Expand_color_map(color_tmp,exp):
    print(color_tmp)
    c_temp= np.zeros((297*exp, 3), np.uint8)
    for i in range(3):
        t = color_tmp[:,i].reshape(297,1)
        c_t= cv2.resize(t, (1,297*exp),  
               interpolation = cv2.INTER_LINEAR )  
        c_temp[:,i]= c_t[:,0]
    return c_temp

path = "20170814.png" # this file is to load the image for color bar

img = cv2.imread(path)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tmp = im_rgb[871:896,573:870]
color_tmp = tmp[3,:,:].astype('float64') 
color_tmp= Expand_color_map(color_tmp, nn)

path = "20160101.png" # to load surface map from sst
img = cv2.imread(path)
# img = img[31:810, 80:869]
img = cv2.resize(img, (120,100), interpolation = cv2.INTER_CUBIC)

def SSH_map(h):
    mina = -1.246199 #min of ssh data
    maxa = 1.232001 #max of ssh data
#    mina = h.min()
#    maxa = h.max()
    temps = np.linspace(mina,maxa,297*nn,endpoint = True)
    m,n = h.shape
    print(m,n)
    SSH_map = np.zeros((m,n, 3), np.float64)
    for i in range(0,m):
        for j in range(n):
            
            if i<768 and j<784 and img[i,j,0]==img[i,j,1]==img[i,j,2]:
                SSH_map[i,j,:] = np.array([180,180,180])
            else:
                idx = np.argmin(np.abs(temps - h[i,j]))
                SSH_map[i,j,:] = color_tmp[idx,:]
#            if h[i,j]==0:
#                SSH_map[i,j,:] = np.array([180,180,180])
#            else:
#                idx = np.argmin(np.abs(temps - h[i,j]))
#                SSH_map[i,j,:] = color_tmp[idx,:]
    return SSH_map




path_save = "ssh_images/"


#path_save = "ssh_images/"
path='/home/devyani/Desktop/download nc files and nc to png/ssh_nc/'
#years = os.listdir(path)
#years=["2005","2006","2007","2008","2009","2010","2011","2012","2013"]
years=["2014","2015","2016","2017","2018","2019","2020"]
#years= ["2015"]#, "2020"]
avg_day=1
for year in years:
    dates_nc= sorted(os.listdir(path+ year))
    date_count = 0
    if not(os.path.exists(path_save + year)):
    	os.mkdir(path_save + year)
    for date in dates_nc:
        # f = Dataset(path+year+"/"+ date, "r")
        ssh1 = NC_image_ssh(path+year+"/"+ date)
        #np.savez(path_save +"/"+ date.split(".")[0]+'.npz',name=sst1)
        
        cc= 1
        for i in range(max(date_count-avg_day+1,0),date_count):
            f = Dataset(path+year+"/"+ dates_nc[i], "r")
            ssh1 = NC_image_ssh(path+year+"/"+ dates_nc[i])
            cc+=1
        ssh1= ssh1/1
        date_count+=1
        
        ssh = SSH_map(ssh1)
        
        ssh = np.uint8(ssh) #, cv2.IMREAD_GRAYSCALE)
        #ssh = cv2.cvtColor(ssh, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_save +year+"/"+ date.split(".")[0]+".png",ssh )
        
        print(date," completed")

