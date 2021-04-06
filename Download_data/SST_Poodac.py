#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:48:57 2020

@author: raghav
"""
'''
Download nc files 
'''

from netCDF4 import Dataset
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from Create_NC import create_NC
path = "20170814.png"

img = cv2.imread(path)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tmp = im_rgb[871:896,573:870]
color_tmp = tmp[3,:,:].astype('float64') 
       


def SSH_map(h):
#    mina = -1.246199 #min of ssh data
#    maxa = 1.232001 #max of ssh data
    h = h-273
    #kalvin to degree
    print(h.max())
    tt= (h ==-6000)*1 + (h ==0)*1
    tt =1*(tt>0)
    hh = h*(1-tt) + h.max()*(tt)
#    mina = hh.min()+260
#    maxa = hh.max()
    mina = 0
    maxa = 32
    print(mina,maxa)
    temps = np.linspace(mina,maxa,297,endpoint = True)
    m,n = h.shape
    SSH_map = np.zeros((m,n, 3), np.float64)
    for i in range(0,m):
        for j in range(n):
            
            if h[i,j]==-6000 or h[i,j]==0 :
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

def NC_to_domain(f):
    
    lon = np.asarray(f.variables["lon"])
    lat = np.asarray(f.variables["lat"])
    
    
    sst =  np.asarray(f.variables["analysed_sst"])
    a= sst[0,:,:]
#    a = np.squeeze(sst, axis=0 )
    
    b =  np.asarray(f.variables["mask"])
    d = ((b!=1)*1)+((a<-3000) *1)+((a>1000)*1)
    d= 1*(d>0)
    d = d[0,:,:]
    b= ((b==2)*1)
    b = b[0,:,:]
    c = np.multiply(a,1-d) + (273-6000)*(b)
    b =  np.asarray(f.variables["mask"])
    b = b[0,:,:]
    domain_lat = [(i,x) for i,x in enumerate(list(lat)) if x>20.0 and x<45.0]
    domain_long = [(i,x) for i,x in enumerate(list(lon)) if x<-55.0 and x>-85.0]
    
    
    domain_map =c[domain_lat[0][0]:domain_lat[-1][0]+1,domain_long[0][0]:domain_long[-1][0]+1]
    b_map =b[domain_lat[0][0]:domain_lat[-1][0]+1,domain_long[0][0]:domain_long[-1][0]+1]
    domain_lat.reverse()
    return list(map(lambda x: x[1],domain_lat)),list(map(lambda x:x[1],domain_long)),np.flip(domain_map, axis = 0),np.flip(b_map, axis = 0)
'''
def NC_image(path):
    f= Dataset(path, "r")
    lon = np.asarray(f.variables["lon"])
    lat = np.asarray(f.variables["lat"])
    
    
    sst =  np.asarray(f.variables["analysed_sst"])
    a= sst[0,:,:]
#    a = np.squeeze(sst, axis=0 )
    
    b =  np.asarray(f.variables["mask"])
    d = ((b!=1)*1)+((a<-3000) *1)+((a>1000)*1)
    d= 1*(d>0)
    d = d[0,:,:]
    b= ((b==2)*1)
    b = b[0,:,:]
    c = np.multiply(a,1-d) + (273-6000)*(b)
#    b =  np.asarray(f.variables["mask"])
#    b = b[0,:,:]
    domain_lat = [(i,x) for i,x in enumerate(list(lat)) if x>20.0 and x<45.0]
    domain_long = [(i,x) for i,x in enumerate(list(lon)) if x<-55.0 and x>-85.0]
    
    
    domain_map =c[domain_lat[0][0]:domain_lat[-1][0]+1,domain_long[0][0]:domain_long[-1][0]+1]
    b_map =b[domain_lat[0][0]:domain_lat[-1][0]+1,domain_long[0][0]:domain_long[-1][0]+1]

    return b_map, np.flip(domain_map, axis = 0)
  '''

'''

path = "/home/raghav/Mtech/LAB/SST_nc_poodac/"
file ="20140101-JPL_OUROCEAN-L4UHfnd-GLOB-v01-fv01_0-G1SST.nc"
file ="20140622-JPL_OUROCEAN-L4UHfnd-GLOB-v01-fv01_0-G1SST.nc"
f = Dataset(path+ file,'r')
#lat = np.asarray(f.variables["lat"])
#lon = np.asarray(f.variables["lon"])
#SST =np.asarray( f.variables["analysed_sst"])
mask =np.asarray( f.variables["mask"])
b, SST = NC_image(path+ file)

sst_map = np.uint8(SSH_map(SST-273))

#plt.imshow(sst_map)
sst_map = cv2.cvtColor(np.uint8(sst_map), cv2.COLOR_RGB2BGR)
sst_re= cv2.resize(sst_map, (788,779),  
               interpolation = cv2.INTER_NEAREST) 
cv2.imwrite(path + file.split("-")[0]+"_1km"+".png",sst_re )

'''


import os 
from datetime import date
from calendar import monthrange
import bz2
#years = [2016","2017","2018"]
years = [ "2007","2008","2009","2010","2011","2012","2013"]

login = "devyani12:nSRK@zXqAkDdmWfgg5@B"
path_mount = "/home/devyani/Desktop/SST_05_13/"
#url = "https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/L4/GLOB/JPL_OUROCEAN/G1SST/"
url = "https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/L4/GLOB/UKMO/OSTIA/"
path_to_save_nc = "sst_nc/"

start_date= ()
end_date = ()
print(years)

for year in years:
    d0 = date(int(year), 1, 1)
    date_no = sorted(os.listdir(path_mount + year))
    yy = int(year)
    if not(os.path.exists(path_to_save_nc + year)):
    	os.mkdir(path_to_save_nc + year)
    for i in range(12):
        _, MM= monthrange(yy, i+1)
        for j in range(MM):
            d1 = date(yy, i+1, j+1)
            delta = d1 - d0
            day_number = delta.days
            folder = os.listdir(path_mount+year+"/"+ date_no[day_number])
            print(folder)
            file1= list(filter(lambda x: x[-3:]=="bz2", folder))
            print(file1)
            file_url = url+ "/".join([year,date_no[day_number], file1[0]])
            print(file_url)
            if os.path.exists(path_to_save_nc +year +"/"+file1[0]):
                os.remove(path_to_save_nc +year +"/"+file1[0])
            if not(os.path.exists(path_to_save_nc+year+ "/" + "".join(str(d1).split("-"))+".nc")):
                fcmd = 'wget --no-verbose --user='+login.split( ":" )[ 0 ]+' --password='+login.split( ":")[ 1 ]+ ' ' +file_url +" -P " + path_to_save_nc +year
                os.system( fcmd )
                file_name = file1[0]
                print(file_name)
                
                with bz2.open(path_to_save_nc +year+"/"+file_name, "rb") as bz:
                    with Dataset('dummy', mode='r', memory=bz.read()) as nc:
                        lat, lon , sst , mask= NC_to_domain(nc)
                        print(sst.shape, "SHAPE")
                        
                        create_NC(path_to_save_nc+year+ "/" + "".join(str(d1).split("-"))+".nc",lat, lon , sst )
                        os.remove(path_to_save_nc +year+"/"+file_name)
                        print(str(d1))
    #                        cv2.imwrite("".join(str(d1).split("-"))+".png",mask*4)
    
            else:
                print(str(d1))



#file1 = "20140223-JPL_OUROCEAN-L4UHfnd-GLOB-v01-fv01_0-G1SST.nc"
#file2= "20140622-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIA.nc"
#h1 = NC_to_domain(Dataset(path+ file1,'r'))
#h4=NC_to_domain(Dataset(path+ file2,'r'))
#cc1= SSH_map(h1[2])
#cc4= SSH_map(h4[2])
#
#
#c_temp= cv2.resize(color_tmp, (3,297*2),  
#               interpolation = cv2.INTER_LINEAR ) 
#cc4 = cv2.cvtColor(np.uint8(cc4), cv2.COLOR_RGB2BGR)
#cv2.imwrite(path + file2.split("-")[0]+"_1km"+".png",cc4 )
