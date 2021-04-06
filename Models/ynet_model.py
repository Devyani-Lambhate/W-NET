import torch
import torch.nn as nn

def conv_block(in_c,out_c):
    conv=nn.Sequential(
        nn.ZeroPad2d((1,1,1,1)),
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.ZeroPad2d((1,1,1,1)),
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
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
        
        
def conv_final_ssh():
    conv=nn.Sequential(
        #nn.Upsample(scale_factor=5, mode='bilinear'),
        nn.Conv2d(1,1,kernel_size=3,padding=1),
        nn.Conv2d(1,1,kernel_size=3,padding=1)
        #conv1 = nn.ConvTranspose2d(1,1, (6, 6), stride=(5, 5),padding=5)
        #conv_final_ssh = MyConvTranspose2d(conv1)
        )
    return conv
    
def conv_final():
    conv=nn.Sequential(
        #nn.Upsample(scale_factor=5, mode='bilinear'),
        nn.Conv2d(1,1,kernel_size=3,padding=1),
        nn.Conv2d(1,1,kernel_size=3,padding=1)
        #conv1 = nn.ConvTranspose2d(1,1, (6, 6), stride=(5, 5),padding=5)
        #conv_final_ssh = MyConvTranspose2d(conv1)
        )
    return conv





def crop_tensor(tensor,target_tensor):
    target_size=target_tensor.size()[2]
    tensor_size=tensor.size()[2]
    delta=tensor_size-target_size
    delta=delta//2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]


    
    
def conv_final_sst():
    conv=nn.Sequential(
        #nn.ZeroPad2d((2,1,2,1)),
        nn.Conv2d(1,1,kernel_size=3,padding=1),
        nn.Conv2d(1,1,kernel_size=3,padding=1)
        )
    return conv



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

        '''

        #encoder
        filter_size=2
        self.down_conv_1=conv_block(3,filter_size*2)
        self.down_conv_2=conv_block(filter_size*2,filter_size*4)
        self.down_conv_3=conv_block(filter_size*4,filter_size*16)
        self.down_conv_4=conv_block(filter_size*16,filter_size*32)
        self.down_conv_5=conv_block(filter_size*32,filter_size*64)

        self.conv_encoder=conv_block(filter_size*128,filter_size*64)
        

        self.relu = nn.ReLU(inplace=False)
        self.max_pool_2x2=nn.MaxPool2d(kernel_size=2)

        #decoder
        self.up_trans_1=nn.ConvTranspose2d(in_channels=filter_size*64,out_channels=filter_size*32,kernel_size=3,stride=2,padding=(1,1))

        self.up_conv_1=conv_block(filter_size*64,filter_size*32)

        self.up_trans_2=nn.ConvTranspose2d(in_channels=filter_size*32,out_channels=filter_size*16,kernel_size=3,stride=2,padding=(1,1))

        self.up_conv_2=conv_block(filter_size*32,filter_size*16)

        self.up_trans_3=nn.ConvTranspose2d(in_channels=filter_size*16,out_channels=filter_size*4,kernel_size=3,stride=2,padding=(1,1))

        self.up_conv_3=conv_block(filter_size*8,filter_size*4)

        self.up_trans_4=nn.ConvTranspose2d(in_channels=filter_size*4,out_channels=filter_size*2,kernel_size=3,stride=2,padding=(1,1))

        self.up_conv_4=conv_block(filter_size*4,filter_size*2)
        
        self.out=nn.Conv2d(
            in_channels=4,
            out_channels=n_classes,
            kernel_size=1
            )


        self.dropout = nn.Dropout(p=0.2)

        self.conv_final=conv_final()
        self.padding = nn.ZeroPad2d((1, 0, 1, 0)) ##LRTB
        '''





    def forward(self,image_sst,image_ssh):


        #############################################"

        #print(image_sst.size(),image_ssh.size())

  

        #encoder SST
        x1_sst=self.down_conv_1(image_sst)
        #print("x1",x1_sst.size())
        x2_sst=self.max_pool_2x2(x1_sst)
        x3_sst=self.down_conv_2(x2_sst)
        #print("x3",x3_sst.size())
        x4_sst=self.max_pool_2x2(x3_sst)
        x4_sst=self.dropout(x4_sst)
        x5_sst=self.down_conv_3(x4_sst)
        #print("x5",x5_sst.size())
        x6_sst=self.max_pool_2x2(x5_sst)
        x6_sst=self.dropout(x6_sst)
        x7_sst=self.down_conv_4(x6_sst)
        #print("x7",x7_sst.size())
        x8_sst=self.max_pool_2x2(x7_sst)
        x8_sst=self.dropout(x8_sst)
        
        x9_sst=self.down_conv_5(x8_sst)
        #print("x9",x7_sst.size())
        x10_sst=self.max_pool_2x2(x9_sst)


        x11_sst=self.down_conv_6(x10_sst)
        #print("x7",x7_sst.size())
        x12_sst=self.max_pool_2x2(x11_sst)

        encoded_sst=self.down_conv_7(x12_sst)
        
     

        #encoder SSH
        x1_ssh=self.down_conv_1(image_ssh)
        x2_ssh=self.max_pool_2x2(x1_ssh)
        x3_ssh=self.down_conv_2(x2_ssh)
        x4_ssh=self.max_pool_2x2(x3_ssh)
        x4_ssh=self.dropout(x4_ssh)
        x5_ssh=self.down_conv_3(x4_ssh)
        x6_ssh=self.max_pool_2x2(x5_ssh)
        x6_ssh=self.dropout(x6_ssh)
        x7_ssh=self.down_conv_4(x6_ssh)
        x8_ssh=self.max_pool_2x2(x7_ssh)
        #x8_ssh=self.dropout(x8_ssh)
        
    
        x8_sst=self.dropout(x8_sst)
        
        x9_ssh=self.down_conv_5(x8_ssh)
        x10_ssh=self.max_pool_2x2(x9_ssh)


        x11_ssh=self.down_conv_6(x10_ssh)
        x12_ssh=self.max_pool_2x2(x11_ssh)

        encoded_ssh=self.down_conv_7(x12_ssh)
    

        ####combine SST and SSH #############

        ###############x1######################
        
        for i in range(4):
            y0=x1_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            #print(y0.size())
            if(i==0):
                x1_sst_conv=o0
            else:
                x1_sst_conv=torch.cat((x1_sst_conv,o0),1)

        #print("x1_sst",x1_ssh.size())

        for j in range(4):
            y1=x1_ssh[:,j:j+1,:,:]
            #print(j,"y0",y1.size())
            o1=self.conv_final(y1)
            if(j==0):
                x1_ssh_conv=o1
            else:
                x1_ssh_conv=torch.cat((x1_ssh_conv,o1),1)

        x1=torch.add(x1_sst_conv,x1_ssh_conv)
        
        #print(x1.size())

        ###############x3######################
        for i in range(8):
            y0=x3_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x3_sst_conv=o0
            else:
                x3_sst_conv=torch.cat((x3_sst_conv,o0),1)

        for i in range(8):
            y0=x3_ssh[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x3_ssh_conv=o0
            else:
                x3_ssh_conv=torch.cat((x3_ssh_conv,o0),1)

        x3=torch.add(x3_sst_conv,x3_ssh_conv)
        
        #print(x3.size())

        ###############x5######################
        for i in range(32):
            y0=x5_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x5_sst_conv=o0
            else:
                x5_sst_conv=torch.cat((x5_sst_conv,o0),1)

        for i in range(32):
            y0=x5_ssh[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x5_ssh_conv=o0
            else:
                x5_ssh_conv=torch.cat((x5_ssh_conv,o0),1)

        x5=torch.add(x5_sst_conv,x5_ssh_conv)
        
        #print('x5',x5.size())



        ###############x7######################
        for i in range(64):
            y0=x7_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x7_sst_conv=o0
            else:
                x7_sst_conv=torch.cat((x7_sst_conv,o0),1)

        for i in range(64):
            y0=x7_ssh[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x7_ssh_conv=o0
            else:
                x7_ssh_conv=torch.cat((x7_ssh_conv,o0),1)

        x7=torch.add(x7_sst_conv,x7_ssh_conv)
        
        #print('x7',x7.size())

        #print(x1.size(),x3.size(),x5.size(),x7.size())
        
        ###############x9######################
        for i in range(128):
            y0=x9_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x9_sst_conv=o0
            else:
                x9_sst_conv=torch.cat((x9_sst_conv,o0),1)

        for i in range(128):
            y0=x9_ssh[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x9_ssh_conv=o0
            else:
                x9_ssh_conv=torch.cat((x9_ssh_conv,o0),1)

        x9=torch.add(x9_sst_conv,x9_ssh_conv)
        
        
        #print('x9',x9.size())

        #print(x1.size(),x3.size(),x5.size(),x7.size())
        
        ###############x11######################
        for i in range(256):
            y0=x11_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x11_sst_conv=o0
            else:
                x11_sst_conv=torch.cat((x11_sst_conv,o0),1)

        for i in range(256):
            y0=x11_ssh[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                x11_ssh_conv=o0
            else:
                x11_ssh_conv=torch.cat((x11_ssh_conv,o0),1)

        x11=torch.add(x11_sst_conv,x11_ssh_conv)

        #print(x1.size(),x3.size(),x5.size(),x7.size())
        
        #print('x11',x11.size())








        ####################################################

        for i in range(512):
            y0=encoded_sst[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                h_sst_conv=o0
            else:
                h_sst_conv=torch.cat((h_sst_conv,o0),1)

        for i in range(512):
            y0=encoded_ssh[:,i:i+1,:,:]
            o0=self.conv_final(y0)
            if(i==0):
                h_ssh_conv=o0
            else:
                h_ssh_conv=torch.cat((h_ssh_conv,o0),1)
      


        #print(h_ssh_conv.size())
        encoded=torch.add(h_ssh_conv,h_sst_conv)
        #print(encoded.size())
        #encoded=self.conv_encoder(encoded)
        #print(encoded.size())


        x13=encoded



   
        x=self.up_trans_1(x13,output_size=x11.shape)
        #x=self.padding(x)
        #print(x.size(),x11.size())
        x=self.up_conv_1(torch.cat([x,x11],1))
        x=self.dropout(x)

        decoder_1=x

        x=self.up_trans_2(x,output_size=x9.shape)
        #x=self.padding(x)
        #print(x.size(),x9.size())
        x=self.up_conv_2(torch.cat([x,x9],1))
        x=self.dropout(x)

        decoder_2=x

        x=self.up_trans_3(x,output_size=x7.shape)
        #x=self.padding(x)
        x=self.up_conv_3(torch.cat([x,x7],1))
        x=self.dropout(x)

        decoder_3=x

        
        x=self.up_trans_4(x,output_size=x5.shape)
        #x=self.padding(x)
        x=self.up_conv_4(torch.cat([x,x5],1))
        x=self.dropout(x)

        decoder_4=x

        x=self.up_trans_5(x,output_size=x3.shape)
        #x=self.padding(x)
        #print(x.size(),x3.size())
        x=self.up_conv_5(torch.cat([x,x3],1))
        x=self.dropout(x)

        decoder_5=x

        x=self.up_trans_6(x,output_size=x1.shape)
        #x=self.padding(x)
        x=self.up_conv_6(torch.cat([x,x1],1))
        x=self.dropout(x)


        result=self.out(x)
        
        y0=result[:,0:1,:,:]
        y1=result[:,1:2,:,:]
        y2=result[:,2:3,:,:]
        y3=result[:,3:4,:,:]
        

        o0=self.conv_final_sst(y0)
        o1=self.conv_final_sst(y1)
        o2=self.conv_final_sst(y2)
        o3=self.conv_final_sst(y3)
        
        result=torch.cat((o0,o1,o2,o3),1)
        #print(h_sst.size())
        #print(h_sst.size())

        
    

        return result ,result,result,result,result,x2_sst,x4_sst,x6_sst,x8_sst,x10_sst,x12_sst,decoder_1,decoder_2,decoder_3,decoder_4,decoder_5,x2_ssh,x4_ssh,x6_ssh,x8_ssh,x10_ssh,x12_ssh,decoder_1,decoder_2,decoder_3,decoder_4,decoder_5


    
#if __name__=="__main__":

#    image=torch.rand(2,1,572,572)
#    model=WNet()
#   print(model(image))




    
    

