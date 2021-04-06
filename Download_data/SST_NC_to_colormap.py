
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:48:57 2020

@author: raghav
"""

'''
to convert nc files to RGB images
'''
nn= 1

from netCDF4 import Dataset
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage import io, exposure, img_as_uint, img_as_float

def values_togreyscale(img):
    mina = img.min()
    maxa = img.max()
    img = 255*(img- mina)/(maxa- mina)
    return np.uint8(img)
    
def to_uint(img):
    mina =img.min()
    maxa = img.max()
    img[:]=[(-x*255+255*mina)/(mina-maxa) for x in img]
    return np.uint8(img)

def Expand_color_map(color_tmp,exp):
    c_temp= np.zeros((297*exp, 3), np.uint8)
    for i in range(3):
        t = color_tmp[:,i].reshape(297,1)
        c_t= cv2.resize(t, (1,297*exp),  
               interpolation = cv2.INTER_LINEAR )  
        c_temp[:,i]= c_t[:,0]
    return c_temp

path = "/home/devyani/Desktop/20170814.png" # this file is to load the image for color bar

img = cv2.imread(path)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tmp = im_rgb[871:896,573:870]
color_tmp = tmp[3,:,:].astype('float64') 
#color_tmp= Expand_color_map(color_tmp, nn)

def SST_map(h):
#    mina = -1.246199 #min of ssh data
#    maxa = 1.232001 #max of ssh data
    h = h-273
    #kalvin to degree
#    print(h.max())
#    tt= (h ==-6000)*1
#    tt =1*(tt>0)
#    hh = h*(1-tt)
#    mina = hh.min()+260
#    maxa = hh.max()
    mina = 0
    
    maxa = 32
#    print(mina,maxa)
    temps = np.linspace(mina,maxa,297*nn, endpoint = True)
    m,n = h.shape
    SSH_map = np.zeros((m,n, 3), np.float64)
    for i in range(0,m):
        for j in range(n):
            
            if h[i,j]==-6000:
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




path = "/home/devyani/Desktop/5.5kmOSTIAdata .nc files/"


path_save = "/home/devyani/Desktop/OSTIAsave/"
#years = os.listdir(path)
years=["2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"]
avg_day=1
count_c=0
for year in years:
    dates_nc= sorted(os.listdir(path+ year))
    date_count = 0
    if not(os.path.exists(path_save + year)):
    	os.mkdir(path_save + year)
    for date in dates_nc:

        f = Dataset(path+year+"/"+ date, "r")
        sst1 = np.asarray(f.variables["Sea Surface Temperature"])
        
        
        
        #print('sst',sst1[300][300])
        #print(sst1)
        '''
        sst1[sst1==-5727.0]=305
        sst1[sst1==-0.0]=305
        print(np.min(sst1),np.max(sst1))

        sst1=np.array(sst1)
        sst1 = values_togreyscale(sst1)
        print(sst1.shape)
        im = Image.fromarray(sst1,mode='L')
        im = im.convert(mode='L')
        if(count_c==0):
            im.save("test_sst.png")
            count_c+=1
        '''

        #print(sst1)
        #np.save(path_save +"/"+ date.split(".")[0]+'.npy',sst1)
        
        cc= 1
        for i in range(max(date_count-avg_day+1,0),date_count):
            f = Dataset(path+year+"/"+ dates_nc[i], "r")
            sst1 = np.asarray(f.variables["Sea Surface Temperature"])
            cc+=1
        sst1= sst1/1
        date_count+=1
        
        
        sst = SST_map(sst1)
        
        sst = np.uint8(sst) #, cv2.IMREAD_GRAYSCALE)
        sst = cv2.cvtColor(sst, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_save +year+"/"+ date.split(".")[0]+".png",sst )
        
        print(date," completed")

