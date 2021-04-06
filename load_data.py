from torchvision.datasets.folder import make_dataset as make_dataset_original
import os
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
from PIL import Image

def make_dataset(root: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    dataset = []

    # Our dir names
    sst_dir = 'SST'
    #ssh_dir='SSH'
    ssh_dir = 'SSH'
    gt_dir = 'GT'   

    # Get all the filenames from RGB folder
    sst_fnames = sorted(os.listdir(os.path.join(root, sst_dir)))
    ssh_fnames = sorted(os.listdir(os.path.join(root, ssh_dir)))

    # Compare file names from GT folder to file names from RGB:
    for gt_fname in sorted(os.listdir(os.path.join(root, gt_dir))):

            if gt_fname in sst_fnames and gt_fname in ssh_fnames:
                # if we have a match - create pair of full path to the corresponding images
                sst_path = os.path.join(root, sst_dir, gt_fname)
                ssh_path = os.path.join(root, ssh_dir, gt_fname)
                gt_path = os.path.join(root, gt_dir, gt_fname)

                item = (sst_path, ssh_path, gt_path)
                # append to the list dataset
                dataset.append(item)
            else:
                continue

    return dataset

#root="/home/devyani/Gulf-Stream-detection/pytorch/scratch/data"
#dataset_original = make_dataset_original(root, {'SST': 0, 'GT': 1}, extensions='png')
#dataset = make_dataset(root)

#print('Original make_dataset:')
#print(*dataset_original, sep='\n')

#print('Our make_dataset:')
#print(*dataset, sep='\n')

def image_to_onehot(mask ,train_on_label,n_classes):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    mask=mask.numpy()
    #print('mask',mask.shape)
    mask=mask.reshape(500,600,1)
    #mask=mask.reshape(512,512,1)
    mask=mask*255
    #print(np.unique(mask))
    #print(mask)
    if(train_on_label=="all"):
        palette=[0, 1, 2, 3]
        #palette=[0,1,2,3]
    elif(train_on_label=="CE"):
        palette=[0,1]
        mask[mask == 2] = 0     # To segment over a particular label like for Cold eddies, make all the pixels that are not cold
        mask[mask == 3] = 0     # eddies as backgound
    elif(train_on_label=="WE"):
        palette=[0,2]
        mask[mask == 1] = 0
        mask[mask == 3] = 0
    elif(train_on_label=="GS"):
        palette=[0,3]
        mask[mask == 1] = 0
        mask[mask == 2] = 0
    
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    #semantic_map=semantic_map.reshape(n_classes,388,388)
    semantic_map=torch.tensor(semantic_map)
    #print('semantic_map',semantic_map.size())
    semantic_map=semantic_map.permute(2,0,1)
    #print(torch.sum(semantic_map,dim=0))
    #print(semantic_map.size())
    return semantic_map

class CustomVisionDataset(VisionDataset):

    def __init__(self,
                 root,
                 train_on_label,
                 n_classes,
                 loader=default_loader,
                 sst_transform=None,
                 gt_transform=None):
        super().__init__(root,
                         transform=sst_transform,
                         target_transform=gt_transform)

        # Prepare dataset
        samples = make_dataset(self.root)

        self.loader = loader
        self.samples = samples
        self.train_on_label=train_on_label
        self.n_classes=n_classes

        # list of SST images
        self.sst_samples = [s[1] for s in samples]
        # list of SSH images
        self.ssh_samples = [s[1] for s in samples]
        # list of GT images
        self.gt_samples = [s[1] for s in samples]
        

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        sst_path, ssh_path, gt_path = self.samples[index]

        # import each image using loader (by default it's PIL)
        sst_sample = self.loader(sst_path)
        ssh_sample = self.loader(ssh_path)
        #print(sst_sample.size)
        gt_sample = self.loader(gt_path)
        #gt_sample = Image.new('I', (500, 600))
        #print(gt_sample.size)




        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        #print('1',sst_sample.shape)
        if self.transform is not None:
            sst_sample = self.transform(sst_sample)
        if self.transform is not None:
            #transform = transforms.Compose([transforms.Resize((100, 120)),transforms.ToTensor()])
            ssh_sample = self.transform(ssh_sample)   
        if self.target_transform is not None:
            gt_sample = self.target_transform(gt_sample)   

        #print('2',sst_sample.size())
        '''


        plot1=sst_sample.numpy()
        #plot1=plot1*50
        
        plot1 = (plot1)*255
        plot1=np.moveaxis(plot1,-1, 0)
        plot1=np.moveaxis(plot1, -1, 1)
        print("@@@@@@", plot1.shape)
        cv2.imwrite( "test_sst_sstitem_2.png", plot1[:, :, : ])

            #plot1=np.moveaxis(plot1,0, -1)
            #plot1=np.moveaxis(plot1, -1, 1)
        #print(plot1.shape)
        '''



    
        # im = Image.fromarray(plot1,mode='RGB')
        # im = im.convert('RGB')
        #im.save("test_sst_gtitem.png")


        # now we return the right imported pair of images (tensors)
        #print("@@@@@@@@@@@@@@@@@@@@",rgb_sample.size(),gt_sample.size())
        #import numpy as np
        #print("gt_sample")
        #x=gt_sample.numpy()
        #print(np.unique(x))
        gt_sample_onehot=image_to_onehot(gt_sample,self.train_on_label,self.n_classes)
        

        return sst_sample, ssh_sample, gt_sample,gt_sample_onehot

    def __len__(self):
        return len(self.samples)



