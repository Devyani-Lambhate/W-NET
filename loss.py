import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous() 

    pred1=pred[:,1:4,:,:]
    target1=target[:,1:4,:,:]
    #pred1=torch.cat((pred[:,0:1,:,:]/100,pred1),1)
    #target1=torch.cat((target[:,0:1,:,:]/100,target1),1)
    #pred1=pred[:,1:4,:,:]
    #target1=target[:,1:4,:,:]

    #print('inside loss',pred1.size(),target1.size())
    intersection = (pred1 * target1).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred1.sum(dim=2).sum(dim=2) + target1.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def dice_loss_weighted(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous() 

    pred1=pred[:,1:2,:,:]*6/13
    target1=target[:,1:2,:,:]*6/13

    pred2=pred[:,2:3,:,:]*6/13
    target2=target[:,2:3,:,:]*6/13

    pred3=pred[:,3:4,:,:]*1/13
    target3=target[:,3:4,:,:]*1/13

    pred_f=torch.cat((pred1,pred2),1)
    pred_f=torch.cat((pred_f,pred3),1)
    target_f=torch.cat((target1,target2),1)
    target_f=torch.cat((target_f,target3),1)

    #print('inside loss',pred1.size(),target1.size())
    intersection = (pred_f * target_f).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred_f.sum(dim=2).sum(dim=2) + target_f.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()
