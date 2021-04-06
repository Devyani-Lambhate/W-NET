import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

def conv_block(in_c,out_c):
    conv=nn.Sequential(
        nn.ZeroPad2d((1,1,1,1)),
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.ZeroPad2d((1,1,1,1)),
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c)
    )
    return conv


def crop_tensor(tensor,target_tensor):
    target_size=target_tensor.size()[2]
    tensor_size=tensor.size()[2]
    delta=tensor_size-target_size
    delta=delta//2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

def conv_final():
    conv=nn.Sequential(
        nn.ZeroPad2d((2,1,2,1)),
        nn.Conv2d(1,1,kernel_size=3),

        nn.Conv2d(1,1,kernel_size=3)
        )
    return conv


def conv_final_sst():
    conv=nn.Sequential(
        #nn.ZeroPad2d((2,1,2,1)),
        nn.Conv2d(1,1,kernel_size=3,padding=1),
        nn.Conv2d(1,1,kernel_size=3,padding=1)
        )
    return conv

def conv_final_ssh():
    conv=nn.Sequential(
        #nn.Upsample(scale_factor=5, mode='bilinear'),
        nn.Conv2d(1,1,kernel_size=3,padding=1),
        nn.Conv2d(1,1,kernel_size=3,padding=1)
        #conv1 = nn.ConvTranspose2d(1,1, (6, 6), stride=(5, 5),padding=5)
        #conv_final_ssh = MyConvTranspose2d(conv1)
        )
    return conv




class MyConvTranspose2d(nn.Module):
    def __init__(self, conv):
        super(MyConvTranspose2d, self).__init__()
#         self.output_size = output_size
        self.conv = conv
        
    def forward(self, x, output_size):
        x = self.conv(x, output_size=output_size)
        return x


def conv_init_ssh():
    #conv1 = nn.ConvTranspose2d(3,3, (6, 6), stride=(5, 5),padding=1)
    #conv_init_ssh = MyConvTranspose2d(conv1)

    conv_init_ssh=nn.Sequential(
    	nn.Upsample(scale_factor=5, mode='bicubic'),
    	
    	)
    return conv_init_ssh
	

def onehot_to_image(onehot_image):
    image=np.zeros((500,600))
    n_classes=2

    for k in range(n_classes):
        onehot_image[k,:,:][onehot_image[k,:,:]==1]=k*50

    if(n_classes==4):
        image=onehot_image[0,:,:]+onehot_image[1,:,:]+onehot_image[2,:,:]+onehot_image[3,:,:]
    elif(n_classes==2):
        image=onehot_image[0,:,:]+onehot_image[1,:,:]
    #print(image)
    return image

class WNet(nn.Module):
    def __init__(self,n_classes,train_on_label):
    
        super(WNet,self).__init__()

        #encoder
        filter_size=2
        self.conv_init_ssh=nn.Upsample(scale_factor=5, mode='bilinear')
        #self.conv_init_ssh=conv_init_ssh()
        self.down_conv_1=conv_block(3,filter_size*2)
        self.down_conv_2=conv_block(filter_size*2,filter_size*4)
        self.down_conv_3=conv_block(filter_size*4,filter_size*16)
        self.down_conv_4=conv_block(filter_size*16,filter_size*32)
        self.down_conv_5=conv_block(filter_size*32,filter_size*64)
        self.down_conv_6=conv_block(filter_size*64,filter_size*128)
        self.down_conv_7=conv_block(filter_size*128,filter_size*256)
        

        self.relu = nn.ReLU(inplace=False)
        self.max_pool_2x2=nn.MaxPool2d(kernel_size=2)

        #decoder
    
        

        up_t1=nn.ConvTranspose2d(in_channels=filter_size*256,out_channels=filter_size*128,kernel_size=3,stride=2,padding=(0,1))
        self.up_trans_1 = MyConvTranspose2d(up_t1)

        self.up_conv_1=conv_block(filter_size*256,filter_size*128)

        up_t2=nn.ConvTranspose2d(in_channels=filter_size*128,out_channels=filter_size*64,kernel_size=3,stride=2,padding=0)
        self.up_trans_2 = MyConvTranspose2d(up_t2)

        self.up_conv_2=conv_block(filter_size*128,filter_size*64)

        up_t3=nn.ConvTranspose2d(in_channels=filter_size*64,out_channels=filter_size*32,kernel_size=3,stride=2,padding=(1,0))
        self.up_trans_3 = MyConvTranspose2d(up_t3)

        self.up_conv_3=conv_block(filter_size*64,filter_size*32)

        up_t4=nn.ConvTranspose2d(in_channels=filter_size*32,out_channels=filter_size*16,kernel_size=3,stride=2,padding=(0,1))
        self.up_trans_4 = MyConvTranspose2d(up_t4)

        self.up_conv_4=conv_block(filter_size*32,filter_size*16)

        up_t5=nn.ConvTranspose2d(in_channels=filter_size*16,out_channels=filter_size*4,kernel_size=3,stride=2,padding=1)
        self.up_trans_5 = MyConvTranspose2d(up_t5)

        self.up_conv_5=conv_block(filter_size*8,filter_size*4)

        up_t6=nn.ConvTranspose2d(in_channels=filter_size*4,out_channels=filter_size*2,kernel_size=3,stride=2,padding=1)
        self.up_trans_6 = MyConvTranspose2d(up_t6)

        self.up_conv_6=conv_block(filter_size*4,filter_size*2)
        
        self.out=nn.Conv2d(
            in_channels=4,
            out_channels=n_classes,
            kernel_size=1
            )


        self.dropout = nn.Dropout(p=0.2)

        self.conv_final=conv_final()
        self.conv_final_sst=conv_final_sst()
        self.conv_final_ssh=conv_final_ssh()
        self.padding = nn.ZeroPad2d((1, 0, 1, 0)) ##LRTB






    def forward(self,image_sst,image_ssh):


        ####################SST#########################
        #print(image_ssh.size(),image_sst.size())

  

        #encoder
        '''
        plot = image_sst.detach().cpu().numpy()
        plot1=plot[0]
        
        plot1=np.moveaxis(plot1, -1, 0)
        plot1=np.moveaxis(plot1, -1, 1)
        print(plot1.shape)

    
        im = Image.fromarray(plot1,mode='RGB')
        im = im.convert('RGB')
        im.save("test_sst_1.png")
        '''
        

        x1=self.down_conv_1(image_sst)
        x2=self.max_pool_2x2(x1)
        #print('down_conv_1',x2.size())

        sst_encoder_1=x2




        x3=self.down_conv_2(x2)
        x4=self.max_pool_2x2(x3)
        #print('down_conv_2',x4.size())

        sst_encoder_2=x4

        x4=self.dropout(x4)
        x5=self.down_conv_3(x4)
        x6=self.max_pool_2x2(x5)
        #print('down_conv_3',x6.size())

        sst_encoder_3=x6

        x6=self.dropout(x6)
        x7=self.down_conv_4(x6)
        x8=self.max_pool_2x2(x7)
        #print('down_conv_4',x8.size())

        sst_encoder_4=x8


        x8=self.dropout(x8)
        x9=self.down_conv_5(x8)
        x10=self.max_pool_2x2(x9)
        #print('down_conv_5',x10.size())
        #x10=self.dropout(x10)

        sst_encoder_5=x10

        x11=self.down_conv_6(x10)
        x12=self.max_pool_2x2(x11)
        #print('down_conv_6',x12.size())
        #x12=self.dropout(x12)
        x13=self.down_conv_7(x12)
        #print('down_conv_7',x13.size())

        sst_encoder_6=x13

        



        #decoder
        
    

        x=self.up_trans_1(x13,output_size=x11.shape)
        #x=self.padding(x)
        #print(x.size(),x11.size())
        x=self.up_conv_1(torch.cat([x,x11],1))
        x=self.dropout(x)

        sst_decoder_1=x

        x=self.up_trans_2(x,output_size=x9.shape)
        #x=self.padding(x)
        #print(x.size(),x9.size())
        x=self.up_conv_2(torch.cat([x,x9],1))
        x=self.dropout(x)

        sst_decoder_2=x

        x=self.up_trans_3(x,output_size=x7.shape)
        #x=self.padding(x)
        x=self.up_conv_3(torch.cat([x,x7],1))
        x=self.dropout(x)

        sst_decoder_3=x

        
        x=self.up_trans_4(x,output_size=x5.shape)
        #x=self.padding(x)
        x=self.up_conv_4(torch.cat([x,x5],1))
        x=self.dropout(x)

        sst_decoder_4=x

        x=self.up_trans_5(x,output_size=x3.shape)
        #x=self.padding(x)
        #print(x.size(),x3.size())
        x=self.up_conv_5(torch.cat([x,x3],1))
        x=self.dropout(x)

        sst_deocder_5=x

        x=self.up_trans_6(x,output_size=x1.shape)
        #x=self.padding(x)
        x=self.up_conv_6(torch.cat([x,x1],1))
        x=self.dropout(x)


        h_sst=self.out(x)
        #print(h_sst.size())
        #print(h_sst.size())

        #################SSH######################
    

        #encoder
        #image_ssh=self.conv_init_ssh(image_ssh)#,output_size=image_sst.shape)
        #print(image_ssh.size())
        #print('image_ssh',image_ssh.size())
        x1=self.down_conv_1(image_ssh)
        x2=self.max_pool_2x2(x1)
        #print('down_conv_1',x2.size())

        ssh_encoder_1=x2

        x3=self.down_conv_2(x2)
        x4=self.max_pool_2x2(x3)
        #print('down_conv_2',x4.size())

        ssh_encoder_2=x4

        x4=self.dropout(x4)
        x5=self.down_conv_3(x4)
        x6=self.max_pool_2x2(x5)
        #print('down_conv_3',x6.size())

        ssh_encoder_3=x6

        x6=self.dropout(x6)
        x7=self.down_conv_4(x6)
        x8=self.max_pool_2x2(x7)
        #print('down_conv_4',x8.size())

        ssh_encoder_4=x8


        x8=self.dropout(x8)
        x9=self.down_conv_5(x8)
        x10=self.max_pool_2x2(x9)
        #print('down_conv_5',x10.size())
        #x10=self.dropout(x10)
        ssh_encoder_5=x10



        x11=self.down_conv_6(x10)
        x12=self.max_pool_2x2(x11)
        #print('down_conv_6',x12.size())
        #x12=self.dropout(x12)
        x13=self.down_conv_7(x12)

        ssh_encoder_6=x13
        



        #decoder
        
        

        x=self.up_trans_1(x13,output_size=x11.shape)
        #x=self.padding(x)
        #print(x.shape,x11.shape)
        x=self.up_conv_1(torch.cat([x,x11],1))
        x=self.dropout(x)

        ssh_decoder_1=x


        x=self.up_trans_2(x,output_size=x9.shape)
        #x=self.padding(x)
        x=self.up_conv_2(torch.cat([x,x9],1))
        x=self.dropout(x)

        ssh_decoder_2=x

        x=self.up_trans_3(x,output_size=x7.shape)
        #x=self.padding(x)
        x=self.up_conv_3(torch.cat([x,x7],1))
        x=self.dropout(x)

        ssh_decoder_3=x

        
        x=self.up_trans_4(x,output_size=x5.shape)
        #x=self.padding(x)
        x=self.up_conv_4(torch.cat([x,x5],1))
        x=self.dropout(x)

        ssh_decoder_4=x

        x=self.up_trans_5(x,output_size=x3.shape)
        #x=self.padding(x)
        #print(x.size(),x3.size())
        x=self.up_conv_5(torch.cat([x,x3],1))
        x=self.dropout(x)

        ssh_decoder_5=x

        x=self.up_trans_6(x,output_size=x1.shape)
        #x=self.padding(x)
        x=self.up_conv_6(torch.cat([x,x1],1))
        x=self.dropout(x)

        h_ssh=self.out(x)
        #print('h_sst', h_sst.shape,h_ssh.shape)

        
        #h_sst=self.relu(h_sst)
        #h_ssh=self.relu(h_ssh)

        #print('h_ssh',h_ssh.size())

        ###############Combining SST and SST################
        
        #h_sst=self.padding(h_sst)
        y0=h_sst[:,0:1,:,:]
        y1=h_sst[:,1:2,:,:]
        y2=h_sst[:,2:3,:,:]
        y3=h_sst[:,3:4,:,:]
        

        o0=self.conv_final_sst(y0)
        o1=self.conv_final_sst(y1)
        o2=self.conv_final_sst(y2)
        o3=self.conv_final_sst(y3)

        h_sst_conv=torch.cat((o0,o1,o2,o3),1)
   

        #h_ssh=self.padding(h_ssh)
        y0=h_ssh[:,0:1,:,:]
        y1=h_ssh[:,1:2,:,:]
        y2=h_ssh[:,2:3,:,:]
        y3=h_ssh[:,3:4,:,:]

        #print(y0.shape)

        o0=self.conv_final_ssh(y0)#,output_size=(500,600))
        #print(o0.shape)
        o1=self.conv_final_ssh(y1)#,output_size=(500,600))
        o2=self.conv_final_ssh(y2)#,output_size=(500,600))
        o3=self.conv_final_ssh(y3)#,output_size=(500,600))

        h_ssh_conv=torch.cat((o0,o1,o2,o3),1)

        #print('h_sst_conv',h_sst_conv.shape,h_ssh_conv.shape)
 

        #h_sst_conv=self.conv_final(h_sst_conv)
        #h_ssh_conv=self.conv_final(h_ssh_conv)

        result=torch.add(h_ssh_conv,h_sst_conv)
        #print(result.size())
        
        #h_sst = F.softmax(h_sst,dim=1)

        #plot = h_sst.detach().cpu().numpy()
        #plot1=plot[0]
        #cv2.imwrite('test_sst.png',plot1)
        '''
        plot1=onehot_to_image(plot1)
        print(plot1.shape)

        plot1=np.moveaxis(plot1, -1, 0)
        plot1=np.moveaxis(plot1, -1, 1)
        
        print(plot1.shape)

        im = Image.fromarray(plot1)
        im = im.convert('RGB')
        im.save("test_sst.png")
        '''


        #h_ssh = F.softmax(h_ssh,dim=1)





        #plot = h_ssh.detach().cpu().numpy()
        #plot1=plot[0]
        #plot1=onehot_to_image(plot1)
        #print(plot1.shape)
        #cv2.imwrite('test_sst.png',plot1)
        '''
        print(plot1.shape)
        plot1=np.moveaxis(plot1, -1, 0)
        plot1=np.moveaxis(plot1, -1, 1)
        

        im = Image.fromarray(plot1)
        im = im.convert('RGB')
        im.save("test_ssh.png")
        '''
        



        #result=h_ssh
        '''

        if(torch.all(torch.eq(h_sst_conv,h_ssh_conv))):
            print("equal")
        else:
            print("not equal")
        '''

        return result ,h_sst_conv,h_ssh_conv,h_sst,h_ssh,sst_encoder_1,sst_encoder_2,sst_encoder_3,sst_encoder_4,sst_encoder_5,sst_encoder_6,sst_decoder_1,sst_decoder_2,sst_decoder_3,sst_decoder_4,sst_deocder_5,ssh_encoder_1,ssh_encoder_2,ssh_encoder_3,ssh_encoder_4,ssh_encoder_5,ssh_encoder_6,ssh_decoder_1,ssh_decoder_2,ssh_decoder_3,ssh_decoder_4,ssh_decoder_5
        #return result,h_sst_conv,h_ssh_conv,h_sst,h_ssh

    
if __name__=="__main__":

    image=torch.rand(2,3,500,600)
    model=WNet(4,'all')
    y=model(image,image)
    print(y.size())
    #image=torch.rand(2,1,512,512)
    #model=WNet(4,'all')
    #y=model(image,image)
    #print(y.size())




    
    

