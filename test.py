from torchsummary import summary
import torch
import torch.nn as nn
import wnet_model_r_1 as wnet_model
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import load_data
from torchvision.transforms import ToTensor
import numpy as np
import PIL
from torchvision import models
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import cv2
import os
import math
import detect_eddies

path='merged/train_data_augmented'
sst_fnames = sorted(os.listdir(os.path.join(path,'SST')))
ssh_fnames = sorted(os.listdir(os.path.join(path, 'SSH')))

# Compare file names from GT folder to file names from RGB:
image_arr=[]
for gt_fname in sorted(os.listdir(os.path.join(path, 'GT'))):
    if gt_fname in sst_fnames and gt_fname in ssh_fnames:
	     image_arr.append(gt_fname)

print(len(image_arr))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_classes=4

'''
LABELS         VALUES

Background  --> 0
Cold eddies --> 1
Warm eddies --> 2
Gulf stream --> 3

n_classes=4 to segment over four labels simulatenously
n_classes=2 to segment over one label at a time
'''
train_on_label="all"
'''
train_on_label="all" for 4 classes

for 2 classes segmentation
if you want to segment gulf stream and background then train_on_label="GS"
if you want to segment cold eddies and background then train_on_label="CE"
if you want to segment warm eddies and background then train_on_label="WE"
'''
batch_size = 8
batch_size_test=16



#####################  Prepare Data #################################################




#interpolation=1 for Nearest Neighbour interpolation
gt_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]) 
sst_transforms = transforms.Compose([transforms.ToTensor()]) 
ssh_transforms = transforms.Compose([transforms.ToTensor()])  # we need this to convert PIL images to Tensor
#dir_train = "merged/train_data_augmented"

# CustomVisionDataset classes has some changes from standard vision dataset class
# The purpose of writing this class is to load in[sst_image, ssh_image, labels, Onehotlabels] 

#train_set = load_data.CustomVisionDataset(dir_train, train_on_label, n_classes, 
#                                        sst_transform=sst_transforms, gt_transform=gt_transforms)
dir_val = "merged/train_data_augmented"
val_set = load_data.CustomVisionDataset(dir_val, train_on_label , n_classes , 
                                        sst_transform=sst_transforms, gt_transform=gt_transforms)
print(val_set)

image_datasets = {
    #'train': train_set,
    'val': val_set
}


dataloaders = {
    #'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=True),#,collate_fn=default_collate),
    'val': DataLoader(val_set, batch_size=batch_size_test, shuffle=False, num_workers=2)#,collate_fn=default_collate)
}

################ load the saved model ######################

device="cuda"
# Model class must be defined somewhere
def init_weights(m):
    if type(m) == nn.Conv2d or type(m)==nn.ConvTranspose2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)

model=wnet_model.WNet(n_classes,train_on_label)
model = nn.DataParallel(model)
model.apply(init_weights)

PATH="wnet-direct-29May-with2019.pth"
model = torch.load(PATH)
#model.load_state_dict(torch.load(PATH))
model=model.to(device)
print("model loaded")


############### Plotting results of validation set ###############################
from PIL import Image
import numpy as np

def onehot_to_image(onehot_image):
    image=np.zeros((500,600))

    for k in range(n_classes):
        onehot_image[k,:,:][onehot_image[k,:,:]==1]=k*50

    if(n_classes==4):
    	image=onehot_image[0,:,:]+onehot_image[1,:,:]+onehot_image[2,:,:]+onehot_image[3,:,:]
    elif(n_classes==2):
    	image=onehot_image[0,:,:]+onehot_image[1,:,:]
    #print(image)
    return image

def to_one_hot(probs):
    
    #probs[0:1,:,:]=0.01
    probs=probs.to(device)
    max_idx = torch.argmax(probs[0:4,:,:], 0, keepdim=True)
    one_hot = torch.FloatTensor(probs.shape)
    one_hot.zero_()
    max_idx=max_idx.to(device)
    one_hot=one_hot.to(device)
    one_hot.scatter_(0, max_idx, 1)
    return one_hot


def plot_images(pred,onehot_labels_f):

    avg_avg_hit_size={'warm': [],
                      'cold': []}
    avg_avg_miss_size={'warm': [],
                        'cold':[]}

    avg_avg_centroid_difference={'warm': [],
                        'cold':[]}    

    avg_avg_area_difference={'warm': [],
                        'cold':[]}    
    avg_avg_diameter_difference={'warm': [],
                        'cold':[]}    


    hit_rate={'warm': [],
              'cold': []}


    hit_rate_75km={'warm': [],
              'cold': []}
 
 
    for l in range(len(pred)):


    
        
        image=onehot_to_image(pred[l])
        #print(onehot_labels_f.shape)
        true_image=onehot_to_image(onehot_labels_f[l])
        d1 = np.abs(image[1:,1:].astype(np.int16) -image[1:,0:-1].astype(np.int16))
        d2 = np.abs(image[1:,1:].astype(np.int16) -image[0:-1,1:].astype(np.int16))
        d=np.maximum(d1,d2)
        d=image[0:499,0:599]
        #print(np.unique(d))
        d[d==50]=51
        d[d==100]=101
        d[d==150]=151
        #print(np.unique(d))

        
        img=np.maximum(d,true_image[0:499,0:599])
        print(img.shape)
        
        dim = np.zeros((499,599))
        #dim=dim*255
        img1 = np.stack((img,dim,dim), axis=2)
        
        #print('img1',img1[50,50,:])

        mask = (img1 == [0.,0.,0.]).all(axis=2)
        img1[mask] =[255,255,255]

        mask = (img1 == [51,0.,0.]).all(axis=2)
        img1[mask] =[0,0,150]
        mask = (img1 == [101.,0.,0.]).all(axis=2)
        img1[mask] =[139,0,0]
        mask = (img1 == [151.,0.,0.]).all(axis=2)
        img1[mask] =[0,128,0]

        mask = (img1 == [50.,0.,0.]).all(axis=2)
        img1[mask] =[173,216,230]
        mask = (img1 == [100.,0.,0.]).all(axis=2)
        img1[mask] =[255,204,203]
        mask = (img1 == [150.,0.,0.]).all(axis=2)
        img1[mask] =[127,255,0]

        
        ####################################
        #######count eddies#################
        true_cold=np.copy(true_image)
        pred_cold=np.copy(image)

        true_warm=np.copy(true_image)
        pred_warm=np.copy(image)

        true_gs=np.copy(true_image)
        pred_gs=np.copy(image)

        true_cold[true_cold==150]=0
        pred_cold[pred_cold==150]=0

        true_warm[true_warm==150]=0
        pred_warm[pred_warm==150]=0
        

        true_cold[true_cold==100]=0
        pred_cold[pred_cold==100]=0

        #print(np.unique(true_warm),np.unique(pred_warm))
       

        true_warm[true_warm==50]=0
        pred_warm[pred_warm==50]=0

        true_warm[true_warm==100]=50
        pred_warm[pred_warm==100]=50
       


        ########## save warm eddies and cold eddies separately ###########

        im = Image.fromarray(true_cold)
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        print('!!!!!!!!',l)
        rgbimg.save('true/cold/'+image_arr[l])
        
        im = Image.fromarray(pred_cold)
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        rgbimg.save("pred/cold/"+image_arr[l])


        im = Image.fromarray(true_warm)
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        rgbimg.save('true/warm/'+image_arr[l])
        
        im = Image.fromarray(pred_warm)
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        rgbimg.save("pred/warm/"+image_arr[l])

        im = Image.fromarray(true_gs)
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        rgbimg.save('true/'+image_arr[l])
        
        im = Image.fromarray(pred_gs)
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        rgbimg.save("pred/"+image_arr[l])
        
        
        eddies=['warm','cold']


        final_img= Image.fromarray(img1.astype('uint8'),mode="RGB")
        draw = ImageDraw.Draw(final_img)



        text=""
        for e in eddies:
        	print(e)
        
	        pred_image=cv2.imread('pred/'+e+'/'+image_arr[l],0)
	        pred_image = pred_image.astype('uint8')
	        nb_components_pred, output_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(pred_image, connectivity=4)

	        true_image=cv2.imread('true/'+e+'/'+image_arr[l],0)
	        true_image = true_image.astype('uint8')
	        nb_components_true, output_true, stats_true, centroids_true = cv2.connectedComponentsWithStats(true_image, connectivity=4)

	        sizes = stats_true[1:, -1]
     
         

        



	        #######################################
	        #-----------removing noise from the mask images by ignoring segments that has less than 10 pixels------------
	        '''

	        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(true_image, connectivity=4)
	        sizes = stats[1:, -1]; nb_components = nb_components - 1
	        img2 = np.zeros((output.shape),dtype = np.uint8)
	        threshold=10
	        #for every component in the image, you keep it only if it's above threshold
	        for i in range(0, nb_components):
	            if sizes[i] >= threshold:
	                img2[output == i + 1] =0

	        nb_components_true, output_true, stats_true, centroids_true = cv2.connectedComponentsWithStats(img2, connectivity=4)
	        '''


	        #######################################
	        #---------------------Label eddies---------------
	        #print(centroids_true)
	        import math
	        h, w = true_image.shape[:2]
	        im_floodfill = true_image.copy()
	        mask = np.zeros((h+2, w+2), np.uint8)
	        u,c=np.unique(true_image,return_counts=True)
	        #print(u,c)

	        #print(true_image.shape)

	        #print('!!!!!!!',true_image[432,51])

	        for j in range(0, len(centroids_true)):
	            #print(math.ceil(centroids_true[j][0]),math.ceil(centroids_true[j][1]))

	            if(true_image[math.ceil(centroids_true[j][1]),math.ceil(centroids_true[j][0])]==0):
	                print(true_image[math.ceil(centroids_true[j][1]),math.ceil(centroids_true[j][0])])
	       
	            else:
	                cv2.floodFill(im_floodfill, mask , (math.ceil(centroids_true[j][0]),math.ceil(centroids_true[j][1])),50+j)
	        im = Image.fromarray(im_floodfill)
	        im.save("test/"+e+'/'+image_arr[l])

	        #######################################
	        '''hit=0



	        for c_t in centroids_true[2:]:
	            for c_p in centroids_pred[2:]:
	                dist=np.linalg.norm(c_t-c_p)
	                if(dist<15):
	                    hit+=1
	                    break
	        '''
            
        
	        #n_eddies,n_eddies_75km,n_correct,avg_hit_size,avg_miss_size,avg_centroid_distance,avg_area_difference,avg_diameter_difference,count_correctly_pred_eddies_diameter_75km=detect_eddies.count_eddies(image_arr[l],e)
	        #print('!!!!!!!!!',n_eddies,n_eddies_75km)
	        #avg_avg_hit_size[e].append(avg_hit_size)
	        #avg_avg_miss_size[e].append(avg_miss_size)
	        #avg_avg_centroid_difference[e].append(avg_centroid_distance)
	        #avg_avg_area_difference[e].append(avg_area_difference)
	        #avg_avg_diameter_difference[e].append(avg_diameter_difference)

	        #hit_rate[e].append(n_correct/n_eddies)

	        #if(n_eddies_75km!=0):
	        #	hit_rate_75km[e].append(count_correctly_pred_eddies_diameter_75km/n_eddies_75km)

            






	        
	        
	        font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
            #n_eddies=0
            #n_correct=0
	        text=text+"number of "+e+" eddies: "+str('n_eddies')+"\ncorrectly predicted: "+str('n_correct')+"\n"
	        
        draw.text((0, 0), text, (0, 0, 0), font=font)
        rgbimg = Image.new("RGB", final_img.size)
        rgbimg.paste(final_img)
        rgbimg.save(image_arr[l])

    

    avg_avg_hit_size_cold=np.array(avg_avg_hit_size['cold'])
    avg_avg_miss_size_cold=np.array(avg_avg_miss_size['cold'])  
    print('Average hit size of cold eddies',np.mean(avg_avg_hit_size_cold))
    print('Average miss size cold eddies',np.mean(avg_avg_miss_size_cold))

    avg_hit_rate_cold=np.array(hit_rate['cold'])
    print('Average hit rate of cold eddies',np.mean(avg_hit_rate_cold))

    avg_hit_rate_warm=np.array(hit_rate['warm'])
    print('Average hit rate of warm eddies',np.mean(avg_hit_rate_warm))

    avg_hit_rate_cold_75km=np.array(hit_rate_75km['cold'])
    print('Average hit rate of cold eddies with diameter more than 75 km',np.mean(avg_hit_rate_cold_75km))

    avg_hit_rate_warm_75km=np.array(hit_rate_75km['warm'])
    print('Average hit rate of warm eddies with diameter more than 75 km',np.mean(avg_hit_rate_warm_75km))






    avg_avg_hit_size_warm=np.array(avg_avg_hit_size['warm'])
    avg_avg_miss_size_warm=np.array(avg_avg_miss_size['warm'])  
    print('Average hit size of warm eddies',np.mean(avg_avg_hit_size_warm))
    print('Average miss size warm eddies',np.mean(avg_avg_miss_size_warm))

    avg_hit_rate_warm=np.array(hit_rate['warm'])
    print('Average hit rate of warm eddies',np.mean(avg_hit_rate_warm))



    avg_avg_centroid_difference_warm=np.array(avg_avg_centroid_difference['warm'])
    avg_avg_centroid_difference_cold=np.array(avg_avg_centroid_difference['cold'])

    print('Average centroid difference of warm eddies',np.mean(avg_avg_centroid_difference_warm))
    print('Average centroid difference of cold eddies',np.mean(avg_avg_centroid_difference_cold))


    avg_avg_area_difference_warm=np.array(avg_avg_area_difference['warm'])
    avg_avg_area_difference_cold=np.array(avg_avg_area_difference['cold'])

    print('Average area difference of warm eddies',np.mean(avg_avg_area_difference_warm))
    print('Average area difference of cold eddies',np.mean(avg_avg_area_difference_cold))


    avg_avg_diameter_difference_warm=np.array(avg_avg_diameter_difference['warm'])
    avg_avg_diameter_difference_cold=np.array(avg_avg_diameter_difference['cold'])

    print('Average area difference of warm eddies',np.mean(avg_avg_diameter_difference_warm))
    print('Average area difference of cold eddies',np.mean(avg_avg_diameter_difference_cold))
    



            
        
       
     

import math


i=1
test_loader=dataloaders["val"]


for sst_inputs,ssh_inputs, labels, onehot_labels in dataloaders["val"]:
    #sst_inputs, ssh_inputs, labels, onehot_labels = next(iter(test_loader))
    #model = nn.DataParallel(model)
    sst_inputs = sst_inputs.to(device)
    ssh_inputs = ssh_inputs.to(device)
    labels = labels.to(device)
    #model.to(device)
    pred_test = model(sst_inputs,ssh_inputs)
    print(pred_test.shape)
    pred_test = F.softmax(pred_test,dim=1)
    if(i==1):
        pred=pred_test
        onehot_labels_f=onehot_labels
    else:
        pred=torch.cat((pred,pred_test), 0)
        onehot_labels_f=torch.cat((onehot_labels_f,onehot_labels),0)
    i=2

print(pred.shape)

avg_true=np.zeros(n_classes)
avg_pred=np.zeros(n_classes)
intersection=np.zeros(n_classes)


for i in range(len(image_arr)):
    '''
    pred2 = pred.detach().cpu().numpy()


    img_plot=pred2[i,0,:,:]+pred2[i,1,:,:]*50+pred2[i,2,:,:]*100+pred2[i,3,:,:]*150
    im = Image.fromarray(img_plot)
    im = im.convert("L")
    im.save("test"+image_arr[i])
    '''



    
    
    #print('labels',onehot_labels_f.shape)
    pred[i]=to_one_hot(pred[i])
    #print('pred',pred[i].shape)

    
    
    pred1=pred[i]
    pred1=onehot_to_image(pred1)
    true=onehot_to_image(onehot_labels_f[i])
    pred1=pred1.detach().cpu().numpy()
    true=true.detach().cpu().numpy()

    multiplied_image=np.multiply(true,pred1)
    #print(multiplied_image.shape)

    for j in range(n_classes):
        avg_true[j] = avg_true[j]+(true == j*50).sum()
        avg_pred[j] = avg_pred[j]+(pred1 == j*50).sum()
        intersection[j]=intersection[j]+(multiplied_image==j*j*50*50).sum()
    
if(train_on_label=="all"):

    print("Cold Eddies accuracy", intersection[1]/(avg_true[1]+avg_pred[1]-intersection[1]))
    print("Warm Eddies accuracy", intersection[2]/(avg_true[2]+avg_pred[2]-intersection[2]))
    print("Gulf Stream accuracy", intersection[3]/(avg_true[3]+avg_pred[3]-intersection[3]))
else:
    print(train_on_label," accuracy", intersection[1]/(avg_true[1]+avg_pred[1]-intersection[1]))


print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if(train_on_label=="all"):

	print("Cold Eddies accuracy", 2*intersection[1]/(avg_true[1]+avg_pred[1]))
	print("Warm Eddies accuracy", 2*intersection[2]/(avg_true[2]+avg_pred[2]))
	print("Gulf Stream accuracy", 2*intersection[3]/(avg_true[3]+avg_pred[3]))
else:
    print(train_on_label," accuracy",2*intersection[1]/(avg_true[1]+avg_pred[1]))


#print(pred[0])
pred = pred.data.cpu().numpy()
onehot_labels_f = onehot_labels_f.data.cpu().numpy()


plot_images(pred,onehot_labels_f)


