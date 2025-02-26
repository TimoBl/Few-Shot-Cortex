import torch
import scipy
import torch.nn as nn
import numpy as np
import torchio as tio
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# for faster training (if hardware allows)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import warnings
from tqdm.notebook import tqdm as tqdm
warnings.filterwarnings('ignore')


# DATA AUGMENTATION
def rotate(img, angle=0):
    return scipy.ndimage.rotate(img, angle, axes=(1, 2), reshape=False,
                                order=3, mode='constant', cval=0.0, prefilter=False)

def get_window(idx, image, shape):

    sx, sy, sz = shp = np.array(shape) // 2
    out = np.ones(tuple(shape)) * image.min() # the background will be the minial value 

    # coordinates in image window
    left_top = (x0, y0, z0) = np.max([idx - shp, [0, 0, 0]], axis=0) # we limit to zero
    right_bottom = (x1, y1, z1) = np.min([idx + shp, image.shape], axis=0) # we limit to border of image

    # coordinates in new image 
    left_top_ = (x0_, y0_, z0_) = left_top - (idx - shp)
    right_bottom_ = (x1_, y1_, z1_) = shape - ((idx + shp) - right_bottom)

    # inpaint
    out[x0_:x1_, y0_:y1_, z0_:z1_] = image[x0:x1+1, y0:y1, z0:z1]

    return out

def inplace_rotation(idx, image, shape, angle):
    # shape of image
    sx = np.array(shape)[0]
    sy, sz = np.array(shape)[1:] // 4
    
    # we get a slightly larger window to avoid artefact
    img = get_window(idx, image, (sx, sy * 6, sz * 6)) # larger crop
    img = rotate(img, angle) # rotate
    img = img[:, sy:-sy, sz:-sz] # crop back
    
    return img


# LOSS FUNCTIONS
class BCE_from_logits(nn.modules.Module):
    def __init__(self):
        super(BCE_from_logits, self).__init__()
    
    def forward(self, input, target):
        #input = input.clamp(min = -1, max = 1)
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        return loss
    
class BCE_from_logits_focal(nn.modules.Module):
    def __init__(self, gamma):
        super(BCE_from_logits_focal, self).__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        #input = input.clamp(min = -1, max = 1)
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        p = input.sigmoid()
        pt = (1 - p) * (1 - target) + p * target
        return ((1 - pt).pow(self.gamma)) * loss
    
# BCE with focal and label weighting
class BCE_from_logits_focal_alpha(nn.modules.Module):
    def __init__(self, gamma, alpha):
        super(BCE_from_logits_focal_alpha, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, input, target):
        #input = input.clamp(min = -1, max = 1)
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        p = input.sigmoid()
        pt = (1 - p) * (1 - target) + p * target
        out = ((1 - pt).pow(self.gamma)) * loss
        return torch.einsum('A,BcAwh->BcAwh', self.alpha, out)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor(alpha) if isinstance(alpha,list) else torch.Tensor([alpha,1-alpha])
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

# CHECKPOINTING
def load_checkpoint(model, path, device, strict=True):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded checkpoint from {}".format(path))
    return checkpoint["epoch"]

def save_checkpoint(model, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }, path)
    print("Saved checkpoint to {}".format(path))