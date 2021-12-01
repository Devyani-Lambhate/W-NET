from torchsummary import summary
import torch
import torch.nn as nn
import wnet_model_HR as wnet_model
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
batch_size = 32
batch_size_test=1

model = wnet_model.WNet(n_classes,train_on_label)
model = model.to(device)

summary(model, input_size=(2, 512, 512))

#####################  Prepare Data #################################################

def default_collate(batch):
    """
    Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512

    We need our own collate function that wraps things up (imge, mask, label).

    In this setup,  batch is a list of tuples (the result of calling: img, mask, label = Dataset[i].
    The output of this function is four elements:
        . data: a pytorch tensor of size (batch_size, c, h, w) of float32 . Each sample is a tensor of shape (c, h_,
        w_) that represents a cropped patch from an image (or the entire image) where: c is the depth of the patches (
        since they are RGB, so c=3),  h is the height of the patch, and w_ is the its width.
        . mask: a list of pytorch tensors of size (batch_size, 1, h, w) full of 1 and 0. The mask of the ENTIRE image (no
        cropping is performed). Images does not have the same size, and the same thing goes for the masks. Therefore,
        we can't put the masks in one tensor.
        . target: a vector (pytorch tensor) of length batch_size of type torch.LongTensor containing the image-level
        labels.
    :param batch: list of tuples (img, mask, label)
    :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
    """
    sst = torch.stack([item[0] for item in batch])
    ssh = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[0] for item in batch])
    onehot_labels = torch.stack([item[0] for item in batch])

    return sst,ssh,labels,onehot_labels


#interpolation=1 for Nearest Neighbour interpolation
gt_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((512, 512)),transforms.ToTensor()]) 
sst_transforms = transforms.Compose([transforms.Resize((512*4, 512*4)),transforms.ToTensor()]) 
ssh_transforms = transforms.Compose([transforms.Resize((512/4, 512/4)),transforms.ToTensor()])  # we need this to convert PIL images to Tensor
dir_train = "merged/train_data_augmented"

# CustomVisionDataset classes has some changes from standard vision dataset class
# The purpose of writing this class is to load in[sst_image, ssh_image, labels, Onehotlabels] 

train_set = load_data.CustomVisionDataset(dir_train, train_on_label, n_classes, 
                                        sst_transform=sst_transforms, gt_transform=gt_transforms)
dir_val = "merged/val_data"
val_set = load_data.CustomVisionDataset(dir_val, train_on_label , n_classes , 
                                        sst_transform=sst_transforms, gt_transform=gt_transforms)
print(train_set,val_set)

image_datasets = {
    'train': train_set, 'val': val_set
}


dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),#,collate_fn=default_collate),
    'val': DataLoader(val_set, batch_size=batch_size_test, shuffle=True, num_workers=2)#,collate_fn=default_collate)
}



###################### Loss Function ###################################################

def calc_loss(pred, target, metrics, bce_weight=0.5):

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    #print(pred,target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

####################### Training function #############################################

def train_model(model, optimizer, num_epochs=120):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                #for param_group in optimizer.param_groups:
                #   print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for sst_inputs,ssh_inputs, labels, onehot_labels in dataloaders[phase]:
                sst_inputs = sst_inputs.to(device)
                ssh_inputs = ssh_inputs.to(device)
                labels = labels.to(device)
                onehot_labels=onehot_labels.to(device)
                
            
                #m=torch.sum(onehot_labels[0],dim=0)
                #print(m.size())
                #print(torch.sum(onehot_labels[0],dim=0))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                #print('sst_inputs',sst_inputs.size())
                with torch.set_grad_enabled(phase == 'train'):
                    #print('!!!!!!!!!!!!!!!!!!!!!!!!',sst_inputs.size())
                    outputs = model(sst_inputs,ssh_inputs)
             
                    loss = calc_loss(outputs, onehot_labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += sst_inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


###########################  Main Training Loop ######################################
import torch.nn as nn

#weight initialization


def init_weights(m):
    if type(m) == nn.Conv2d or type(m)==nn.ConvTranspose2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = wnet_model.WNet(n_classes,train_on_label).to(device)
model = nn.DataParallel(model)
model.apply(init_weights)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=60, gamma=0.1)

model = train_model(model, optimizer_ft, num_epochs=120)


################## save the model #########################
path="WNET-HR.pth"
torch.save(model.module.state_dict(), path)
print("model saved")

'''
############### Plotting results of validation set ###############################
from PIL import Image
import numpy as np

def onehot_to_image(onehot_image):
    image=np.zeros((512,512))

    for k in range(n_classes):
        onehot_image[k,:,:][onehot_image[k,:,:]==1]=k*50

    
    image=onehot_image[0,:,:]+onehot_image[1,:,:]+onehot_image[2,:,:]+onehot_image[3,:,:]
    #print(image)
    return image

def to_one_hot(probs):
    #print(probs.size())
    probs=probs.to(device)
    max_idx = torch.argmax(probs, 0, keepdim=True)
    one_hot = torch.FloatTensor(probs.shape)
    one_hot.zero_()
    max_idx=max_idx.to(device)
    one_hot=one_hot.to(device)
    one_hot.scatter_(0, max_idx, 1)
    return one_hot


def plot_images(pred):
    for l in range(len(pred)):
        image=onehot_to_image(pred[l])
        im = Image.fromarray(image)
        im = im.convert("L")
        im.save(str(l)+".png")

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


for i in range(69):
    pred[i]=to_one_hot(pred[i])
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


print("Cold Eddies accuracy", 2*intersection[1]/(avg_true[1]+avg_pred[1]))
print("Warm Eddies accuracy", 2*intersection[2]/(avg_true[2]+avg_pred[2]))
print("Gulf Stream accuracy", 2*intersection[3]/(avg_true[3]+avg_pred[3]))

#print(pred[0])
pred = pred.data.cpu().numpy()


plot_images(pred)
'''
