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
from PIL import Image
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device="cpu"
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
batch_size = 16
batch_size_test= 16

model = wnet_model.WNet(n_classes,train_on_label)
model = model.to(device)

#summary(model, input_size=(2, 512, 512))

#####################  Prepare Data #################################################



#interpolation=1 for Nearest Neighbour interpolation
gt_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((500, 600)),transforms.ToTensor()]) 
sst_transforms = transforms.Compose([transforms.ToTensor()]) 
#ssh_transforms = transforms.Compose([transforms.Resize((512/4, 512/4)),transforms.ToTensor()])  # we need this to convert PIL images to Tensor
#gt_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((500, 600)),transforms.ToTensor()]) 
#sst_transforms = transforms.Compose([transforms.ToTensor()]) 
#ssh_transforms = transforms.Compose([transforms.ToTensor()])  # we need this to convert PIL images to Tensor
dir_train = "merged/train_data_augmented"
dir_val = "merged/val_data"
dir_test = "merged/test_data"

# CustomVisionDataset classes has some changes from standard vision dataset class
# The purpose of writing this class is to load in[sst_image, ssh_image, labels, Onehotlabels] 

train_set = load_data.CustomVisionDataset(dir_train, train_on_label, n_classes, 
                                        sst_transform=sst_transforms, gt_transform=gt_transforms)


val_set = load_data.CustomVisionDataset(dir_val, train_on_label, n_classes, 
                                        sst_transform=sst_transforms, gt_transform=gt_transforms)



test_set = load_data.CustomVisionDataset(dir_test, train_on_label , n_classes , 
                                        sst_transform=sst_transforms, gt_transform=gt_transforms)
print('!!!!!!!!!!!!!!!!!!!!!!!')
print(len(train_set),len(val_set),len(test_set))

image_datasets = {
    'train': train_set, 'val': val_set,'test' :test_set
}


dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(val_set, batch_size=batch_size_test, shuffle=False, num_workers=2),
    'test': DataLoader(test_set, batch_size=batch_size_test, shuffle=True, num_workers=2)
}



###################### Loss Function ###################################################

def calc_loss(pred, target, metrics, bce_weight=1):

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

from matplotlib import pyplot as plt

###################################################

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

def onehot_to_image(onehot_image):
    image=np.zeros((500,600))
    #image=np.zeros((512,512))


    for k in range(n_classes):
        onehot_image[k,:,:][onehot_image[k,:,:]==1]=k*50

    if(n_classes==4):
    	image=onehot_image[0,:,:]+onehot_image[1,:,:]+onehot_image[2,:,:]+onehot_image[3,:,:]
    if(n_classes==2):
    	image=onehot_image[0,:,:]+onehot_image[1,:,:]

    #print(image)
    return image

def matplotlib_imshow(img, one_channel=False):
    #print(img)
    #if one_channel:
    #    img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(outputs):

	pred= F.softmax(outputs,dim=1)
	#pred = pred.data.cpu().numpy()
	#print(pred.shape)

	fig = plt.figure(figsize=(12, 48))
	for idx in np.arange(3):

	    pred[idx]=to_one_hot(pred[idx])
	    image = pred[idx].data.cpu().numpy()
	    image=onehot_to_image(image)


	    ax = fig.add_subplot(1, 3, idx+1, xticks=[], yticks=[])
	    matplotlib_imshow(image, one_channel=True)
        #ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
        #    classes[preds[idx]],
        #    probs[idx] * 100.0,
        #    classes[labels[idx]]),
        #            color=("green" if preds[idx]==labels[idx].item() else "red"))

	return fig
###################################################


#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/train_val_2')

def train_model(model, optimizer, num_epochs=60):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_loss_arr=[]
    val_loss_arr=[]

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
                #print(sst_inputs[0].shape)
                #plot1=sst_inputs[0].numpy()

                #plot1=np.moveaxis(plot1, -1, 0)
                #plot1=np.moveaxis(plot1, -1, 1)
                #print(plot1.shape)

    
                #im = Image.fromarray(plot1,mode='RGB')
                #im = im.convert('RGB')
                #im.save("test_sst.png")
                
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
                    outputs,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_= model(sst_inputs,ssh_inputs)
                    #outputs,_,_,_,_=model(sst_inputs,ssh_inputs)

                    #print('outputs',outputs.size())
             
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

            if phase == 'train' :

                train_loss_arr.append(metrics['loss']/epoch_samples)


                #writer.add_scalar('training loss',
                #            metrics['loss']/epoch_samples,
                #            epoch +1)

            if phase == 'val' :

                val_loss_arr.append(metrics['loss']/epoch_samples)


                #writer.add_scalar('validation loss',
                #            metrics['loss']/epoch_samples,
                #           epoch +1)


        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        #outputs = model(sst_inputs,ssh_inputs)
        #print('outputs',outputs.size())

        
        #writer.add_figure('output',
        #                   plot_classes_preds(outputs),
        #                    global_step=epoch +1)




    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_loss_arr,val_loss_arr


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

model,train_loss_arr,val_loss_arr = train_model(model, optimizer_ft, num_epochs=200)

plt.grid()
plt.plot(train_loss_arr)
plt.plot(val_loss_arr)
np.savetxt('train_loss.txt',train_loss_arr, delimiter =" ", fmt="%s") 
np.savetxt('val_loss.txt',val_loss_arr, delimiter =" ", fmt="%s") 
plt.title('Train and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('train_val_loss.png')

################## save the model #########################
path="wnet-direct-7-27march-bce.pth"
torch.save(model, path)
path="wnet-statedict-7-27march-bce.pth"
torch.save(model.state_dict(), path)
print("model saved")

############### Plotting results of validation set ###############################


from PIL import Image
import numpy as np




def plot_images(plot_image,which):
	if (which=='pred'):
	    for l in range(len(plot_image)):
	        image=onehot_to_image(plot_image[l])
	        im = Image.fromarray(image)
	        im = im.convert("L")
	        im.save(str(l)+which+".png")
	else:
		for l in range(len(plot_image)):
			#print(pred.shape)
			image0=plot_image[l][0]
			image1=plot_image[l][1]
			image2=plot_image[l][2]
			image3=plot_image[l][3]
			im = Image.fromarray(image0)
			im = im.convert("L")
			im.save(str(l)+which+"channel0"+".png")

			im = Image.fromarray(image1)
			im = im.convert("L")
			im.save(str(l)+which+"channel1"+".png")

			im = Image.fromarray(image2)
			im = im.convert("L")
			im.save(str(l)+which+"channel2"+".png")

			im = Image.fromarray(image3)
			im = im.convert("L")
			im.save(str(l)+which+"channel3"+".png")


import math


i=1
test_loader=dataloaders["val"]

for sst_inputs,ssh_inputs, labels, onehot_labels in dataloaders["val"]:
    #sst_inputs, ssh_inputs, labels, onehot_labels = next(iter(test_loader))
    #print(sst_inputs.shape)



    sst_inputs = sst_inputs.to(device)
    ssh_inputs = ssh_inputs.to(device)
    labels = labels.to(device)
    pred_test, h_sst_conv, h_ssh_conv,h_sst, h_ssh,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(sst_inputs,ssh_inputs)
    #pred_test, h_sst_conv, h_ssh_conv,h_sst, h_ssh = model(sst_inputs,ssh_inputs)

    h_sst=h_sst_conv
    h_ssh=h_ssh_conv

    print(pred_test.shape)
    pred_test = F.softmax(pred_test,dim=1)
    if(i==1):
        pred=pred_test
        onehot_labels_f=onehot_labels
        h_sst_f=h_sst
        h_ssh_f=h_ssh

        h_sst_conv_f=h_sst_conv
        h_ssh_conv_f=h_ssh_conv
    else:
        pred=torch.cat((pred,pred_test), 0)
        onehot_labels_f=torch.cat((onehot_labels_f,onehot_labels),0)
        h_sst_f=torch.cat((h_sst_f,h_sst),0)
        h_ssh_f=torch.cat((h_ssh_f,h_ssh),0)

        h_sst_conv_f=torch.cat((h_sst_conv_f,h_sst_conv),0)
        h_ssh_conv_f=torch.cat((h_ssh_conv_f,h_ssh_conv),0)
    i=2

print(pred.shape)

avg_true=np.zeros(n_classes)
avg_pred=np.zeros(n_classes)
intersection=np.zeros(n_classes)


for i in range(19):
    pred[i]=to_one_hot(pred[i])
    pred1=pred[i]
    pred1=onehot_to_image(pred1)
    true=onehot_to_image(onehot_labels_f[i])
    pred1=pred1.detach().cpu().numpy()
    true=true.detach().cpu().numpy()

    h_sst_f[i]=to_one_hot(h_sst_f[i]) 
    h_ssh_f[i]=to_one_hot(h_ssh_f[i])

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
	print(train_on_label," accuracy",2* intersection[1]/(avg_true[1]+avg_pred[1]))


#print(pred[0])
pred = pred.data.cpu().numpy()
h_sst_f = h_sst_f.data.cpu().numpy()
h_ssh_f = h_ssh_f.data.cpu().numpy()

plot_images(pred,'pred')
plot_images(h_sst_f,'sst')
plot_images(h_ssh_f,'ssh')

plot_images(h_sst_conv_f,'sst_conv_')
plot_images(h_ssh_conv_f,'ssh_conv_')



