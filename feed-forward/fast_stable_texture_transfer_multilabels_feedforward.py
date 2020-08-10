"""
semantic neural style transfer with content loss and style loss
this is for multiple labels in masks--the first stage of DeepPhotoStyle

So far, this works for normal feed-forward neural style transfer and 
semantic neural style transfer with masks input.

Inputs: one content image ( or a batch of images for training) and a few style
        images with corresponding masks.

Notice: the labeled areas are as large as possible since smaller regions won't 
        be learned very well

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import sys
import re

#from collections import namedtuple
import math
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import pytorch_colors as colors
import copy 
import argparse
import operator
from torchvision import datasets
from transformer_net import TransformerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#---------utils functions------------------------------------------------

def load_image(filename, mask=False, size=None, scale=None, square=False):

    img = Image.open(filename)
    if not square: 
        if size is not None:
            w,h = img.size
            if w > size:
                img = img.resize( (size, int(math.ceil(size * h / w)) ), Image.ANTIALIAS)
            elif h > size:
                img = img.resize( (int(math.ceil(size * w / h)), size ), Image.ANTIALIAS)
        if scale is not None:       
            img = img.resize( (int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    else:
        img = img.resize((size,size), Image.ANTIALIAS)

    loader = transforms.ToTensor()
                                    
    img = loader(img)
    if not mask: 
        img = img[:3].unsqueeze(0).to(device, torch.float)
    else:
        img = img[:1].unsqueeze(0).to(device, torch.float)
    return img


def original_colors(content, output):

    output_y = colors.rgb_to_yuv(output.data)[0][0].unsqueeze(0)
    content_uv = colors.rgb_to_yuv(content.data)[0][1:]#.unsqueeze(0)
    result = torch.cat((output_y, content_uv),0)
    result = colors.yuv_to_rgb(result)
    result = result.unsqueeze(0)
    #result.data.clamp_(0,1)
    return result                    



def save_mask(filename, data):
    unloader = transforms.ToPILImage()
    image = data.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(filename)


def preprocess(img):
    #convert RGB to BGR and substract the mean values
    mean_pixel = torch.Tensor([[[103.939]], [[116.779]], [[123.68]]]).to(device, torch.float)
    #print('img shape: ', img.shape)
    permute = [2,1,0]
    img = img[:,permute,:,:].mul(255.0)
    mean_pixel = mean_pixel.unsqueeze(0).repeat(1,1,1,1).expand_as(img)
    img = img - mean_pixel
    return img

def deprocess(img):
    mean_pixel = torch.Tensor([[[103.939]], [[116.779]], [[123.68]]]).to(device, torch.float)
    mean_pixel = mean_pixel.unsqueeze(0).repeat(1,1,1,1).expand_as(img)
    img = img + mean_pixel
    permute = [2,1,0]
    img = img[:,permute,:,:].div(255.0)
    return img


def save_image(filename, data):
    #img = data.clone().clamp(0, 255).numpy()
    data = deprocess(data)
    data.clamp_(0,1)
    unloader = transforms.ToPILImage()
    image = data.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(filename)
    


# utilize the LBFGS optimizer
def get_input_optimizer(input_img, args, model):
    # this line to show that input is a parameter that requires a gradient
    if args.optimizer == 'lbfgs':
        print('Using L-BFGS optimizer...')
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif args.optimizer == 'adam':
        print('Using Adam optimizer...')
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    return optimizer

def extract_mask(seg_ori, color, mask_type):
    #['blue', 'green', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
  if mask_type == 'single':
    mask = None
    seg = seg_ori.squeeze()
    if color == 'blue':
        mask = torch.lt(seg[0], 0.1)
        mask *= torch.lt(seg[1], 0.1)
        mask *= torch.gt(seg[2], 1-0.1)
    elif color == 'green':
        mask = torch.lt(seg[0], 0.1)
        mask *= torch.gt(seg[1], 1-0.1)
        mask *= torch.lt(seg[2], 0.1)

    elif color == 'white':
        mask = torch.gt(seg[0], 1-0.1)
        mask *= torch.gt(seg[1], 1-0.1)
        mask *= torch.gt(seg[2], 1-0.1)
    elif color == 'red':
        mask = torch.gt(seg[0], 1-0.1)
        mask *= torch.lt(seg[1], 0.1)
        mask *= torch.lt(seg[2], 0.1)
    elif color == 'yellow':
        mask = torch.gt(seg[0], 1-0.1)
        mask *= torch.gt(seg[1], 1-0.1)
        mask *= torch.lt(seg[2], 0.1)
    elif color == 'grey':
        mask = torch.mul(  torch.gt(seg[0], 0.5-0.1), torch.lt(seg[0], 0.5+0.1) )
        mask *= torch.mul( torch.gt(seg[1], 0.5-0.1), torch.lt(seg[1], 0.5+0.1) )
        mask *= torch.mul( torch.gt(seg[2], 0.5-0.1), torch.lt(seg[2], 0.5+0.1) )
    elif color == 'lightblue':
        mask = torch.lt(seg[0], 0.1)
        mask *= torch.gt(seg[1], 1-0.1)
        mask *= torch.gt(seg[2], 1-0.1)
    elif color == 'purple':
        mask = torch.gt(seg[0], 1-0.1)
        mask *= torch.lt(seg[1], 0.1)
        mask *= torch.gt(seg[2], 1-0.1)
    else:
        print(' extract_mask: not recognized color, color = ', color)

    return mask.float().unsqueeze(0)

  elif mask_type == 'set':
    mask = None
    idx = None
    #seg = seg_ori.squeeze()
    if color == 'blue':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze() 
            mask = torch.lt(seg[0], 0.1)
            mask *= torch.lt(seg[1], 0.1)
            mask *= torch.gt(seg[2], 1-0.1)
            if mask.sum() != 0 :
                idx = j
                break
    elif color == 'green':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze() 
            mask = torch.lt(seg[0], 0.1)
            mask *= torch.gt(seg[1], 1-0.1)
            mask *= torch.lt(seg[2], 0.1)
            if mask.sum() != 0 :
                idx = j
                break

    elif color == 'white':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze() 
            mask = torch.gt(seg[0], 1-0.1)
            mask *= torch.gt(seg[1], 1-0.1)
            mask *= torch.gt(seg[2], 1-0.1)
            if mask.sum() != 0 :
                idx = j
                break
    elif color == 'red':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze()
            mask = torch.gt(seg[0], 1-0.1)
            mask *= torch.lt(seg[1], 0.1)
            mask *= torch.lt(seg[2], 0.1)
            if mask.sum() != 0 :
                idx = j
                break
    elif color == 'yellow':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze()
            mask = torch.gt(seg[0], 1-0.1)
            mask *= torch.gt(seg[1], 1-0.1)
            mask *= torch.lt(seg[2], 0.1)
            if mask.sum() != 0 :
                idx = j
                break
    elif color == 'grey':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze()
            mask = torch.mul(  torch.gt(seg[0], 0.5-0.1), torch.lt(seg[0], 0.5+0.1) )
            mask *= torch.mul( torch.gt(seg[1], 0.5-0.1), torch.lt(seg[1], 0.5+0.1) )
            mask *= torch.mul( torch.gt(seg[2], 0.5-0.1), torch.lt(seg[2], 0.5+0.1) )
            if mask.sum() != 0 :
                idx = j
                break
    elif color == 'lightblue':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze()
            mask = torch.lt(seg[0], 0.1)
            mask *= torch.gt(seg[1], 1-0.1)
            mask *= torch.gt(seg[2], 1-0.1)
            if mask.sum() != 0 :
                idx = j
                break
    elif color == 'purple':
        for j in range(len(seg_ori)):
            seg = seg_ori[j].squeeze()
            mask = torch.gt(seg[0], 1-0.1)
            mask *= torch.lt(seg[1], 0.1)
            mask *= torch.gt(seg[2], 1-0.1)
            if mask.sum() != 0 :
                idx = j
                break
    else:
        print(' extract_mask: not recognized color, color = ', color)

    return mask.float().unsqueeze(0), idx

#---------Loss functions-----------------------------------------------------
class TVLoss(nn.Module):
    
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
    
    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


class ContentLoss(nn.Module):
    def __init__(self, weight, mode=None, requires_grad=False):
        super(ContentLoss, self).__init__()
        self.target = torch.Tensor()
        self.weight = weight        
        self.mode = mode
        self.loss = 0       
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, input):
        self.input = input
        if self.mode == 'capture':
             self.target = input.detach()
        elif self.mode == 'loss':
             self.loss = F.mse_loss(input, self.target) * self.weight
        return input
"""
#for batch_size = 1
def gram_matrix(input):
    a, b, c, d = input.size()  # a = batch size (=1)
    # b = number of feature maps
    # (c, d) = dimensions of a f. map (N=c*d)

    features = input.view(a*b, c*d)
    features_t = features.transpose(1, 0)
    G = torch.mm(features, features_t)  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c *d)
"""
#for batch_size > 1
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a, b, c*d)
    features_t = features.transpose(1, 2)
    G = features.bmm(features_t)  # compute the gram product

    return G.div(b * c *d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature, weight, mode=None):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.weight = weight
        self.mode = mode
        self.loss = 0
        if self.mode == 'capture':
            self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        self.input = input      
        if self.mode == 'loss':
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target) * self.weight
        return input

class StyleLoss_Seg(nn.Module):

    def __init__(self, target_feature, weight, mode=None, input_masks=None, 
                                       target_masks=None, color_codes=None):
        super(StyleLoss_Seg, self).__init__()
        self.weight = weight
        self.mode = mode
        
        self.input_masks = []
        for mk in range(len(input_masks)):
            self.input_masks.append(input_masks[mk].detach())
        
        self.target_masks = []
        for tk in range(len(target_masks)):
            self.target_masks.append(target_masks[tk].detach())

        self.color_codes = color_codes
        self.target = []
        if self.mode == 'capture':
           for j in range(len(self.color_codes)):
               target_msk = self.target_masks[j].expand_as(target_feature[j])              
               target_masked = torch.mul(target_feature[j], target_msk)
               target_msk_mean = torch.mean(self.target_masks[j])
               target_local = gram_matrix(target_masked).detach()
               if target_msk_mean > 0:
                   target_local.div(target_feature[j].nelement() * target_msk_mean) 
               self.target.append(target_local)
   
    def forward(self, input):

        self.input = input

        if self.mode == 'loss':
            self.loss = 0         

            #print('target_mask.shape', self.target_masks[1].shape)
            #print('input_mask.shape', self.input_masks[1].shape) 
            for j in range(len(self.color_codes)):

                
                #save_mask('target_mask3.png', self.target_masks[2])
                #save_mask('input_mask3.png', self.input_masks[2])
                #assert 0 == 1
                input_msk = self.input_masks[j].expand_as(input)
                input_masked = torch.mul(input, input_msk)
                input_msk_mean = torch.mean(self.input_masks[j])
                input_local_G = gram_matrix(input_masked)
                if input_msk_mean > 0:
                    input_local_G.div(input.nelement() * input_msk_mean)
                loss_local = F.mse_loss(input_local_G, self.target[j])
                loss_local *= self.weight * input_msk_mean
                self.loss += loss_local
        return input
    

#--------------build loss net---------------------------------------------------------------------
def build_loss_model(cnn, args, style_image_sets, color_codes=None, 
                          color_content_masks=None, color_style_masks=None): 
   
    cnn = copy.deepcopy(cnn)
    
    content_losses = []
    style_losses = []
   
    model = nn.Sequential()
    if args.tv_weight > 0:
        tv_loss = TVLoss(args.tv_weight)
        model.add_module("tv_loss", tv_loss)
    i = -1 
    for layer in cnn.children():
        
        i += 1        

        if isinstance(layer, nn.Conv2d):            
            name = 'conv_{}'.format(i)
            
            if args.semantic == 1:
                tmp_net = nn.AvgPool2d(3,1,1).to(device)
                for j in range(len(color_codes)):
                    color_content_masks[j] = tmp_net(color_content_masks[j].repeat(1,1,1,1)).detach()
                    color_style_masks[j] = tmp_net(color_style_masks[j].repeat(1,1,1,1)).detach()
            if str(i) in args.content_layers:
                 for stl in style_losses:
                     stl.mode = None
                 print ('setting up content layers...')
                 #target = model(content_image).detach()
                 content_loss = ContentLoss(args.content_weight)
                 model.add_module("content_layer_{}".format(i), content_loss)
                 content_losses.append(content_loss)
                 

            if str(i) in args.style_layers:
                 for ctl in content_losses:
                     ctl.mode = None
                 print ('capturing style target...')
                 
                 if args.semantic == 1:
                     target = []
                     for m in range(len(style_image_sets)):
                         target.append(model(style_image_sets[m]).detach())
                     style_loss = StyleLoss_Seg(target, args.style_weights[0], 'capture', 
                                                color_content_masks.copy(), 
                                                color_style_masks.copy(),
                                                color_codes )                   
                 elif args.semantic == 0:
                     target = model(style_image_sets).detach()
                     style_loss = StyleLoss(target, args.style_weights[0], 'capture')
                 
                 model.add_module("style_layer_{}".format(i), style_loss)
                 style_losses.append(style_loss)
                 
           
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            if args.pooling == 'avg':
                print('replacing the maxpooling with avgpooling...')
                k_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
                layer = nn.AvgPool2d(k_size, stride, padding).to(device)

            if str(i) in args.content_layers:
                 for stl in style_losses:
                     stl.mode = None
                 print ('setting up content layer...')
                 #target = model(content_image).detach()
                 content_loss = ContentLoss(args.content_weight)
                 model.add_module("content_layer_{}".format(i), content_loss)
                 content_losses.append(content_loss)
                
            if str(i) in args.style_layers:
                 for ctl in content_losses:
                     ctl.mode = None
                 print ('capturing style target...')                
                 if args.semantic == 1:
                     target = []
                     for m in range(len(style_image_sets)):
                         target.append(model(style_image_sets[m]).detach())
                     style_loss = StyleLoss_Seg(target, args.style_weights[0], 'capture', 
                                                color_content_masks.copy(), 
                                                color_style_masks.copy(),
                                                color_codes )
                 elif args.semantic == 0:
                     target = model(style_image_sets).detach()
                     style_loss = StyleLoss(target, args.style_weights[0] , 'capture')

                 model.add_module("style_layer_{}".format(i), style_loss)
                 style_losses.append(style_loss)
                 
            if args.semantic == 1 and color_content_masks is not None:
                for j in range(len(color_codes)):
                    #print(color_content_masks[j].shape)
                    color_content_masks[j] = F.interpolate(color_content_masks[j], 
                                size=[int(math.floor(color_content_masks[j].shape[2]/2)),
                                      int(math.floor(color_content_masks[j].shape[3]/2))],
                                mode='bilinear',
                                align_corners=False)
            if args.semantic == 1 and color_style_masks is not None:
                for j in range(len(color_codes)):
                    color_style_masks[j] = F.interpolate(color_style_masks[j], 
                                size=[int(math.floor(color_style_masks[j].shape[2]/2)), 
                                      int(math.floor(color_style_masks[j].shape[3]/2))],
                                mode='bilinear',
                                align_corners=False)  

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

    for i in range(len(model) - 1, -1, -1):
      if args.semantic == 1:
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss_Seg):
            break
      else:
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]
    #print(model)
    
    # set mode = 'loss' to style_loss and content_loss modules
    for stl in style_losses:
        stl.mode = 'None'
    for ctl in content_losses:
        ctl.mode = 'None'
    
    return model, content_losses, style_losses


#----------run train-------------------------------------------------------------------------------

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train_preparation_mask(args):

    style_image_sets = []       
    style_image_split = args.style_images.split(',')
    #print(style_image_split)
    for style in style_image_split:
        print('style image:', style)
        style_image = load_image(style,mask=False, 
                             size=args.image_size, scale=args.style_scale)        
            
        style_image = preprocess(style_image)
        style_image = style_image.repeat(args.batch_size, 1,1,1)                        
        style_image_sets.append(style_image)

    content_mask_image = load_image(args.content_mask_image, 
                                        mask=False, size=args.image_size, scale=None, square=True) 

    style_mask_sets = []
    style_mask_split = args.style_mask_images.split(',')
    for mask in style_mask_split:
        print('style mask image: ', mask)
        style_mask_image = load_image(mask, mask=False, 
                                      size=args.image_size, scale=args.style_scale)  
        style_mask_sets.append(style_mask_image)
    del mask, style_mask_image

    cnn = None
    if args.loss_model == 'vgg19':
        cnn = models.vgg19(pretrained=True).features.to(device).eval()  
    elif args.loss_model == 'vgg16': 
        cnn = models.vgg16(pretrained=True).features.to(device).eval()

    color_codes = ['blue', 'green', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    color_content_masks, color_style_masks, style_image_sets_real, color_codes_real = [], [], [], []
        
    for j in range(len(color_codes)):
        content_mask_j = extract_mask(content_mask_image, color_codes[j], mask_type='single')
        style_mask_j, idx = extract_mask(style_mask_sets, color_codes[j], mask_type='set')
        if content_mask_j.sum() != 0 and style_mask_j.sum() != 0:               
            color_codes_real.append(color_codes[j])
            #print('idx : ',idx)
            color_content_masks.append(content_mask_j.unsqueeze(0))
            color_style_masks.append(style_mask_j.unsqueeze(0))
            style_image_sets_real.append(style_image_sets[idx])  
    print('color codes are: ')
    print(color_codes_real)

    del j
    # get tranform net, content losses, style losses
    loss_net, content_losses, style_losses = build_loss_model(cnn, 
                                                args, style_image_sets_real.copy(), color_codes_real,
                                                color_content_masks.copy(), color_style_masks.copy())

    #collect space back 
    cnn = None
    del cnn
    
    
    # cat content_masks into [args.batch_size, n_guidances, size, size]
    k = 0
    while k < len(color_codes_real):
           if k == 0:
              content_masks = torch.cat((color_content_masks[k].squeeze(0), color_content_masks[k+1].squeeze(0)), 0)
              k = 2
           else:
              content_masks = torch.cat( (content_masks, color_content_masks[k].squeeze(0)), 0)
              k+=1
    content_masks = content_masks.unsqueeze(0).repeat(args.batch_size,1,1,1)
    del k

    n_channels = len(color_codes_real) + 3

    return loss_net, content_losses, style_losses, content_masks, n_channels

def train_preparation(args):

    style_image = load_image(args.style_images,mask=False, 
                             size=args.image_size, scale=args.style_scale)
    style_image = preprocess(style_image)   
    style_image = style_image.repeat(args.batch_size, 1,1,1)

    cnn = None
    if args.loss_model == 'vgg19':
        cnn = models.vgg19(pretrained=True).features.to(device).eval()  
    elif args.loss_model == 'vgg16': 
        cnn = models.vgg16(pretrained=True).features.to(device).eval()

    loss_net, content_losses, style_losses = build_loss_model(cnn,
                                                 args, style_image)

    n_channels = 3

    return loss_net, content_losses, style_losses, n_channels

#--------------------------------training stage-----------------------------------------------------

def run_train(args):
 

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('running training process...')
    if args.semantic == 1:
        print('multilabels semantic feedforward neural style transfer training...')
    elif args.semantic == 0:
        print('normal feedforward neural style transfer training...')
    
    

    if args.semantic == 1:
        loss_net, content_losses, style_losses, content_masks, n_channels = train_preparation_mask(args)
    elif args.semantic == 0:
        loss_net, content_losses, style_losses, n_channels = train_preparation(args)


    
    if args.backend == 'cudnn':
       torch.backends.cudnn.enabled=True


    transform = transforms.Compose([
                           transforms.Resize(args.image_size),
                           transforms.CenterCrop(args.image_size),
                           transforms.ToTensor(),                          
                           ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transform_net = TransformerNet(n_channels).to(device)
    mse_loss = nn.MSELoss()

   
    optimizer = optim.Adam(transform_net.parameters(), lr=args.learning_rate)

    iteration = [0]
    while iteration[0] <= args.epochs-1:
        transform_net.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            stloss = 0.
            ctloss = 0.
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            

            #stack color_content_masks into x as input
            x, x_ori = x.to(device), x.to(device).clone()
            x = preprocess(x)
            x_ori = preprocess(x_ori)
            if args.semantic == 1:
                x = torch.cat((x, content_masks),1)
            
            y = transform_net(x)

            #compute pixel loss
            if args.semantic == 1:
                y_pix = torch.cat((y, content_masks), 1)
            elif args.semantic == 0:
                y_pix = y
            pixloss = 0. 
            if args.pixel_weight > 0:
               pixloss = mse_loss(x, y_pix) * args.pixel_weight

            #compute content loss and style loss
            
            for ctl in content_losses:
                ctl.mode = 'capture'
            loss_net(x_ori)
            for ctl in content_losses:
                ctl.mode = 'loss'
            for stl in style_losses:
                stl.mode = 'loss'
            loss_net(y)
            for ctl in content_losses:          
                ctloss += mse_loss(ctl.input, ctl.target) * args.content_weight
            if args.semantic == 1:
              for stl in style_losses:
                for u in range(len(stl.color_codes)):
                    input_msk = stl.input_masks[u].expand_as(stl.input)
                    input_masked = torch.mul(stl.input, input_msk)
                    input_msk_mean = torch.mean(stl.input_masks[u])
                    input_local_G = gram_matrix(input_masked)
                    if input_msk_mean > 0:
                        input_local_G.div(stl.input.nelement() * input_msk_mean)
                    loss_local = mse_loss(input_local_G, stl.target[u])
                    loss_local *=  input_msk_mean
                    #larger target areas multiples smaller style weight
                    if input_msk_mean > 0.2:
                        stloss += loss_local *args.style_weights[0]
                    #smaller target areas multiples larger style weight
                    elif input_msk_mean <= 0.2:
                        #print('aaaaa')
                        stloss += loss_local *args.style_weights[1]
            elif args.semantic == 0:
              for stl in style_losses:
                  gram = gram_matrix(stl.input)
                  stloss += mse_loss(gram, stl.target) * args.style_weights[0]          

            loss = ctloss + stloss + pixloss
            
            loss.backward()
            optimizer.step()
            
            agg_content_loss += ctloss.item()
            agg_style_loss += stloss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}, Epoch {}:\t[{}/{}], content: {:.6f}, style: {:.6f}, total: {:.6f}".format(
                                             time.ctime(), iteration[0] , count, len(train_dataset),
                                             agg_content_loss / (batch_id + 1),
                                             agg_style_loss / (batch_id + 1),
                                             (agg_content_loss + agg_style_loss) / (batch_id + 1)
                                             )
                print(mesg)
            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transform_net.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(iteration[0]+1) + "_batch_id_" + str(
                                      batch_id + 1) + "_semantic_" + str(args.semantic) +  ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transform_net.state_dict(), ckpt_model_path)
                transform_net.to(device).train()

               
        iteration[0] += 1
                
           
    #save final model
    transform_net.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(
                          ' ', '_') + "_content_" + str(args.content_weight) + "_style_" + str(
                          args.style_weights[0]) + "_semantic_" + str(
                          args.semantic) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transform_net.state_dict(), save_model_path)

    print("\n training process is Done!, trained model saved at", save_model_path)    
        
  
def run_feedforward_stylization(args):

    content_image = load_image(args.content_image, 
                                        mask=False, size=args.image_size, scale=None, square=False)   
  

    content_image = preprocess(content_image)
    print('content image shape:', content_image.shape)


    if args.semantic == 1:
        content_mask_image = load_image(args.content_mask_image, 
                                        mask=False, size=args.image_size, scale=None, square=False)  


        style_mask_sets = []
        style_mask_split = args.style_mask_images.split(',')
        for mask in style_mask_split:
            print('style mask image: ', mask)
            style_mask_image = load_image(mask, mask=False, 
                                          size=args.image_size, scale=args.style_scale)  
            style_mask_sets.append(style_mask_image)

        color_codes = ['blue', 'green', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
        color_content_masks, color_style_masks, style_image_sets_real, color_codes_real = [], [], [], []
        
        for j in range(len(color_codes)):
            content_mask_j = extract_mask(content_mask_image, color_codes[j], mask_type='single')
            style_mask_j, idx = extract_mask(style_mask_sets, color_codes[j], mask_type='set')
            if content_mask_j.sum() != 0 and style_mask_j.sum() != 0:               
                color_codes_real.append(color_codes[j])
                #print('idx : ',idx)
                color_content_masks.append(content_mask_j.unsqueeze(0))
                color_style_masks.append(style_mask_j.unsqueeze(0))
        print('color codes are: ')
        print(color_codes_real)
        del j

        # cat content_masks into [1, n_guidances, size, size]
        j = 0
        while j < len(color_codes_real)-1:
               if j == 0:
                  content_masks = torch.cat((color_content_masks[j].squeeze(0), color_content_masks[j+1].squeeze(0)), 0)
                  j = 2
               else:
                  content_masks = torch.cat( (content_masks, color_content_masks[j].squeeze(0)), 0)
                  j+=1
        del j
        print('content masks shape', content_masks.shape)
        content_masks = content_masks.unsqueeze(0).repeat(1,1,1,1)
        #save_mask('color_content_masks_grey.png', color_content_masks[3])

        input_image = torch.cat((content_image, content_masks),1)

        n_channels = len(color_codes_real)-1 + 3

    elif args.semantic == 0:
        content_mask_image =None

        style_mask_image = None

        input_image = content_image

        n_channels = 3

  
    style_model = TransformerNet(n_channels)
    state_dict = torch.load(args.style_model)
    
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\/running_(mean|var)$', k):
            del state_dict[k]
    del k
    style_model.load_state_dict(state_dict)
    style_model = style_model.eval().to(device)

    start_time = time.time()
    output = style_model(input_image)
    print('run time : {:4f} s'.format(time.time() - start_time))

    if args.original_colors == 1:
        output = original_colors(content_image,output)    
    save_image(filename=args.output_image, data=output.detach()) 
    
#---------------add arguments for inputs-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='parser for traing feedforwad stable texture transfer')
     
    parser.add_argument("--training", type=int, default=0,
                        help="0 is for stylization, and 1 is for training")
    # options for training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, 
                        default='../Fast-neural-style-transfer/images/coco/',
                        help="path to the training folder which contains all training images")
    parser.add_argument("--save_model_dir", type=str, default='models/saved/butterfly/')
    parser.add_argument("--checkpoint_model_dir", type=str, default='models/checkpoints/butterfly/')
    parser.add_argument("--style_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)

    # options for layers
    parser.add_argument("--content_layers", type=list, default=['9'])
    parser.add_argument("--style_layers", type=list,
                        default=['4','9','16','23'])
    

    # options for weights
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument("--style_weights", type=float, default=[1e2,5e2]) #1e10,5e10
    parser.add_argument("--pixel_weight", type=float, default=0)
    parser.add_argument("--tv_weight", type=float, default=-1)
    # options for images
    parser.add_argument("--style_images", type=str, required=False,
                        help="please add the path to your style image.")
    parser.add_argument("--content_mask_image", type=str, required=False,
                        help="please add the path to your content mask image.")
    parser.add_argument("--style_mask_images", type=str, required=False,
                        help="please add the path to your style mask image.")

    # alternative options
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--semantic", type=int, default=1)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--backend", choices=['nn','cudnn'], default='cudnn')
    parser.add_argument("--optimizer", choices=['lbfgs','adam'], default='adam')
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--pooling", choices=['max', 'avg'], default='max')
    parser.add_argument("--loss_model", type=str, default='vgg16')
    parser.add_argument("--style_scale", type=float, default=1)    

    #options for stylization
    parser.add_argument("--content_image", type=str, default="./data/multilabels/44.png")
    parser.add_argument("--output_image", type=str, default="./data/multilabels/44_stylized.png")
    parser.add_argument("--style_model", type=str, default='models/checkpoints/mix/ckpt_epoch_1_batch_id_2000_semantic_1.0.pth')
    parser.add_argument('--original_colors', type=int, default=1)
    args = parser.parse_args()

    start_time = time.time()
    if args.training == 1:
       run_train(args)
       print('run time : {:4f} h'.format((time.time() - start_time)/3600))
    else:
       run_feedforward_stylization(args)
       
                 


if __name__ == "__main__":
    main()


