import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

base_model = models.resnet18(pretrained=False)
    
list(base_model.children())


########################################################33

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def conv_final():
    conv=nn.Sequential(
        nn.ZeroPad2d((1,0,1,0)),
        nn.Conv2d(1,1,kernel_size=3)
        )
    return conv


class WNet(nn.Module):

    def __init__(self,n_classes,train_on_label):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=False)
        
        self.base_layers = list(base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  

        self.layer5 = self.base_layers[8]  # size=(N, 512, x.H/32, x.W/32)
        self.layer5_1x1 = convrelu(1024, 1024, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up4 = convrelu(512+1024, 1024, 3,1)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_classes, 1)
        self.conv_final=conv_final()
        self.padding = nn.ZeroPad2d((1, 0, 1, 0)) ##LRTB
        
    def forward(self, image_sst,image_ssh):
    
        ################## SST ##########################
        #print("image_sst",image_sst.size())
        x_original = self.conv_original_size0(image_sst)
        #print("x_original_sst",x_original.size())
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(image_sst)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        
        layer5 = self.layer5_1x1(layer5)
        x = self.upsample(layer5)

        layer4 = self.layer4_1x1(layer3)
        x = torch.cat([x, layer4], dim=1)
        x = self.conv_up4(x)
 

        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        h_sst = self.conv_last(x)    
        
        ############### SSH #############################
        
        x_original = self.conv_original_size0(image_ssh)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(image_ssh)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        
        layer5 = self.layer5_1x1(layer5)
        x = self.upsample(layer5)

        layer4 = self.layer4_1x1(layer3)
        x = torch.cat([x, layer4], dim=1)
        x = self.conv_up4(x)
        
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        h_ssh = self.conv_last(x)  
        
        ###############Combining SST and SST################
        h_sst=self.padding(h_sst)
        y0=h_sst[:,0:1,:,:]
        y1=h_sst[:,1:2,:,:]
        y2=h_sst[:,2:3,:,:]
        y3=h_sst[:,3:4,:,:]
        
        o0=self.conv_final(y0)
        o1=self.conv_final(y1)
        o2=self.conv_final(y2)
        o3=self.conv_final(y3)

        h_sst_conv=torch.cat((o0,o1,o2,o3),1)
   

        h_ssh=self.padding(h_ssh)
        y0=h_ssh[:,0:1,:,:]
        y1=h_ssh[:,1:2,:,:]
        y2=h_ssh[:,2:3,:,:]
        y3=h_ssh[:,3:4,:,:]

        o0=self.conv_final(y0)
        o1=self.conv_final(y1)
        o2=self.conv_final(y2)
        o3=self.conv_final(y3)

        h_ssh_conv=torch.cat((o0,o1,o2,o3),1)
 

        #h_sst_conv=self.conv_final(h_sst_conv)
        #h_ssh_conv=self.conv_final(h_ssh_conv)

        result=torch.add(h_ssh_conv,h_sst_conv)    
    
          
        
        return result


     
# check keras-like model summary using torchsummary

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WNet(4,"all")
model = model.to(device)

summary(model, input_size=(2,3, 512, 512))



