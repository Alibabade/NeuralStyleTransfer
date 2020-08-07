"""
semantic neural style transfer with content loss and style loss
this is for multiple labels in masks--the first stage of DeepPhotoStyle

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

#from collections import namedtuple
import math
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import pytorch_colors as colors
import copy 
import argparse
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#---------utils functions------------------------------------------------

def load_image(filename, mask=False, size=None, scale=None):

    img = Image.open(filename)
    if size is not None:
        w,h = img.size
        if w > size:
            img = img.resize( (size, int(math.ceil(size * h / w)) ), Image.ANTIALIAS)
        elif h > size:
            img = img.resize( (int(math.ceil(size * w / h)), size ), Image.ANTIALIAS)
        
    if scale is not None:
        
        img = img.resize( (int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
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
    #result.data.clamp_(0,255.0)
    return result                    


def save_image(filename, data):
    data = deprocess(data)
    data.clamp_(0,1)
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
    img = img[:,permute,:,:].mul(256.0)
    mean_pixel = mean_pixel.unsqueeze(0).repeat(1,1,1,1).expand_as(img)
    img = img - mean_pixel
    return img

def deprocess(img):
    mean_pixel = torch.Tensor([[[103.939]], [[116.779]], [[123.68]]]).to(device, torch.float)
    mean_pixel = mean_pixel.unsqueeze(0).repeat(1,1,1,1).expand_as(img)
    img = img + mean_pixel
    permute = [2,1,0]
    img = img[:,permute,:,:].div(256.0)
    return img

def normalize_img(img):
    # normalize using imagenet mean and std
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (img - mean) / std


class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1,1,1)
        self.std = std.clone().detach().view(-1,1,1)
        
    def forward(self, img):
        return (img - self.mean) / self.std

# utilize the LBFGS optimizer
def get_input_optimizer(input_img, args, model):
    # this line to show that input is a parameter that requires a gradient
    if args.optimizer == 'lbfgs':
        print('Using L-BFGS optimizer...')
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif args.optimizer == 'adam':
        print('Using Adam optimizer...')
        optimizer = optim.Adam([input_img.requires_grad_()], lr=args.learning_rate)
    return optimizer


def extract_mask(seg_ori, color):
    #['blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
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
    elif color == 'black':
        mask = torch.lt(seg[0], 0.1)
        mask *= torch.lt(seg[1], 0.1)
        mask *= torch.lt(seg[2], 0.1)
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

#---------Loss functions-----------------------------------------------------

class TVLoss(nn.Module):
    
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
    
    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + \
                                     torch.sum(torch.abs(self.y_diff)))
        return input

   

class ContentLoss(nn.Module):
    def __init__(self, target_feature, weight, mode=None):
        super(ContentLoss, self).__init__()
        self.target = torch.Tensor()
        self.weight = weight        
        self.mode = mode
        self.loss = 0
        if self.mode == 'capture':
             self.target = target_feature.detach()
    def forward(self, input):
        if self.mode == 'loss':
             self.loss = F.mse_loss(input, self.target) * self.weight
        return input



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

class StyleLoss(nn.Module):

    def __init__(self, target_feature, weight, mode=None, input_mask=None, target_mask=None):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.weight = weight
        self.mode = mode
        self.loss = 0           
        self.input_mask = input_mask.detach() if input_mask is not None else None
        self.target_mask = target_mask.detach() if target_mask is not None else None
        if self.mode == 'capture':
            
            if self.target_mask is not None:                
                target_msk = self.target_mask.clone().expand_as(target_feature)                           
                target_local = torch.mul(target_feature, target_msk)                                
                target_msk_mean = torch.mean(target_msk)                
                self.target = gram_matrix(target_local).detach()
                if target_msk_mean > 0 :                
                    self.target.div(target_feature.nelement() * target_msk_mean)
            else:                
                self.target = gram_matrix(target_feature).detach()


    def forward(self, input):
        if self.mode == 'loss':
         
            if self.input_mask is not None:
                
                input_msk = self.input_mask.clone().expand_as(input)
                input_local = torch.mul(input, input_msk)
                input_msk_mean = torch.mean(input_msk)                
                self.G = gram_matrix(input_local)#.detach()                
                if input_msk_mean > 0:
                    self.G.div(input.nelement()*input_msk_mean)
                self.loss = F.mse_loss(self.G, self.target) 
                self.loss = self.loss * self.weight
            else:                
                self.G = gram_matrix(input)
                self.loss = F.mse_loss(self.G, self.target) * self.weight

        return input

class StyleLoss_Seg(nn.Module):

    def __init__(self, target_feature, weight, mode=None, input_masks=None, 
                                       target_masks=None, color_codes=None):
        super(StyleLoss_Seg, self).__init__()
        self.weight = weight
        self.mode = mode
        self.input_masks = input_masks.detach()
        self.target_masks = target_masks.detach()
        self.color_codes = color_codes
        self.target = []
        if self.mode == 'capture':
           for j in range(len(self.color_codes)):
               target_msk = self.target_masks[j].expand_as(target_feature)
               target_masked = torch.mul(target_feature, target_msk)
               target_msk_mean = torch.mean(self.target_masks[j])
               target_local = gram_matrix(target_masked).detach()
               if target_msk_mean > 0:
                   target_local.div(target_feature.nelement() * target_msk_mean) 
               self.target.append(target_local)
    def forward(self, input):
        if self.mode == 'loss':
            self.loss = 0
            for j in range(len(self.color_codes)):
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
    

#--------------build transform net---------------------------------------------------------------------
def build_transform_model(cnn, args, content_image, style_image, color_codes, 
                          color_content_masks, color_style_masks): 
   
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
                 print ('capturing content target...')
                 target = model(content_image).detach()
                 content_loss = ContentLoss(target, args.content_weight, 'capture' )
                 model.add_module("content_layer_{}".format(i), content_loss)
                 content_losses.append(content_loss)

            if str(i) in args.style_layers:
                 for ctl in content_losses:
                     ctl.mode = None
                 print ('capturing style target...')
                 target = model(style_image).detach()
                 if args.semantic == 1:
                     style_loss = StyleLoss_Seg(target, args.style_weight, 'capture', 
                                                torch.stack(color_content_masks), 
                                                torch.stack(color_style_masks),
                                                color_codes )
                 elif args.semantic == 0:
                     style_loss = StyleLoss(target, args.style_weight, 'capture')
                 
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
            if color_content_masks is not None:
                for j in range(len(color_codes)):
                    #print(color_content_masks[j].shape)
                    color_content_masks[j] = F.interpolate(color_content_masks[j], 
                                size=[int(math.floor(color_content_masks[j].shape[2]/2)),
                                      int(math.floor(color_content_masks[j].shape[3]/2))],
                                mode='bilinear',
                                align_corners=False)
            if color_style_masks is not None:
                for j in range(len(color_codes)):
                    color_style_masks[j] = F.interpolate(color_style_masks[j], 
                                size=[int(math.floor(color_style_masks[j].shape[2]/2)), 
                                      int(math.floor(color_style_masks[j].shape[3]/2))],
                                mode='bilinear',
                                align_corners=False)


            if str(i) in args.content_layers:
                 for stl in style_losses:
                     stl.mode = None
                 print ('capturing content target...')
                 target = model(content_image).detach()
                 content_loss = ContentLoss(target, args.content_weight, 'capture' )
                 model.add_module("content_layer_{}".format(i), content_loss)
                 content_losses.append(content_loss)

            if str(i) in args.style_layers:
                 for ctl in content_losses:
                     ctl.mode = None
                 print ('capturing style target...')
                 target = model(style_image).detach()
                 if args.semantic == 1:
                     style_loss = StyleLoss_Seg(target, args.style_weight, 'capture', 
                                                torch.stack(color_content_masks), 
                                                torch.stack(color_style_masks),
                                                color_codes )
                 elif args.semantic == 0:
                     style_loss = StyleLoss(target, args.style_weight, 'capture')
                 
                 model.add_module("style_layer_{}".format(i), style_loss)
                 style_losses.append(style_loss)

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss_Seg):
            break
    model = model[:(i+1)]
    #print(model)

    # set mode = 'loss' to style_loss and content_loss modules
    for stl in style_losses:
        stl.mode = 'loss'
    for ctl in content_losses:
        ctl.mode = 'loss'

    return model, content_losses, style_losses, tv_loss


#----------run texture transfer-----------------------------------------------------------------------
def run_texture_transfer(args):

    print('running neural style transfer...')
    
    content_image = load_image(args.content_image,mask=False, 
                               size=args.image_size, scale=None)
    content_image = preprocess(content_image)
    style_image = load_image(args.style_image,mask=False, 
                             size=args.image_size, scale=args.style_scale)
    style_image = preprocess(style_image)
    

    if args.content_mask_image:
        content_mask_image = load_image(args.content_mask_image, 
                                        mask=False, size=args.image_size, scale=None)  
    else:
        content_mask_image =None
    if args.style_mask_image:
        style_mask_image = load_image(args.style_mask_image, mask=False, 
                                      size=args.image_size, scale=args.style_scale)  
    else: 
        style_mask_image = None

    cnn = None
    if args.model == 'vgg19':
        cnn = models.vgg19(pretrained=True).features.to(device).eval()  
    elif args.model == 'vgg16': 
        cnn = models.vgg16(pretrained=True).features.to(device).eval()
   
    
    #extract semantic areas from mask images
    color_codes = ['blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    #color_codes = ['blue', 'green',  'black', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    color_content_masks, color_style_masks, color_codes_real = [], [], []
    for j in range(len(color_codes)):
        content_mask_j = extract_mask(content_mask_image, color_codes[j])
        style_mask_j = extract_mask(style_mask_image, color_codes[j])
        if content_mask_j.sum() != 0 and style_mask_j.sum() != 0:
            color_codes_real.append(color_codes[j])
            color_content_masks.append(content_mask_j.unsqueeze(0))
            color_style_masks.append(style_mask_j.unsqueeze(0))
    print('color codes are: ')
    print(color_codes_real)
    #print(color_content_masks[-1].shape)
    # get tranform net, content losses, style losses
    transform_net, content_losses, style_losses, tv_loss = build_transform_model(cnn, 
                                                args, content_image, style_image, color_codes_real,
                                                color_content_masks, color_style_masks)


    #collect space back 
    cnn = None
    del cnn


    if args.backend == 'cudnn':
       torch.backends.cudnn.enabled=True



    # initilize the input image
    initial_image = None
    if args.init_image == 'random':
        initial_img = torch.randn(content_image.data.size(), device=device)
    elif args.init_image == 'image':
        initial_img = content_image.clone()
    input_img = initial_img
    optimizer = get_input_optimizer(input_img, args, transform_net)

    print('start optimizing process ...')
    iteration = [0]
    while iteration[0] <= args.max_nums:
        
        def feval():
            #input_img.data.clamp_(0, 255.0)
            optimizer.zero_grad()
            transform_net(input_img)
            stloss = 0
            ctloss = 0
            tvloss = tv_loss.loss
            for stl in style_losses:
                stloss += stl.loss
            for ctl in content_losses:
                ctloss += ctl.loss
            loss = stloss + ctloss + tvloss
            
            loss.backward()
            iteration[0] += 1
            
            if iteration[0] % 100 == 0:
                print("    optimizing iters: {:d}".format(iteration[0]))
                print("        content loss   : {:4f}".format(ctloss.item()))
                print("        style loss     : {:4f}".format(stloss.item()))
                print("        tv loss        : {:4f}".format(tvloss.item()))
                print("        total loss     : {:4f}".format(loss.item()))

                output_name = args.output_image[:-4] + '_' + str(iteration[0]) + args.output_image[-4:] 
                output = input_img.clone()
                #output.data.clamp_(0,255.0)
                if args.original_colors == 1:
                    output = original_colors(content_image,output)
                if iteration[0] % args.save_iter == 0:                  
                    save_image(filename=output_name, data=output)
                
            return loss
        optimizer.step(feval)
        
    #input_img.data.clamp_(0,255.0)
    output = input_img.clone()
    if args.original_colors == 1:
        output = original_colors(content_image,output) 
    return output
    
#---------------add arguments for inputs-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='parser for fast stable texture transfer')
    # options for images
    parser.add_argument("--content_image", type=str, required=True,
                        help="please add the path to your conent image.")
    parser.add_argument("--style_image", type=str, required=True,
                        help="please add the path to your style image.")
    parser.add_argument("--content_mask_image", type=str, required=False,
                        help="please add the path to your content mask image.")
    parser.add_argument("--style_mask_image", type=str, required=False,
                        help="please add the path to your style mask image.")
    # options for layers
    parser.add_argument("--content_layers", type=list, default=['23'])
    parser.add_argument("--style_layers", type=list,
                        default=['2','7','12','21','30'])
    parser.add_argument("--hist_layers", type=list, 
                        default=['2','21'])
    # options for weights
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument("--style_weight", type=float, default=1e10)
    parser.add_argument("--hist_weight", type=float, default=1)
    parser.add_argument("--tv_weight", type=float, default=1e-3)
    # alternative options 
    parser.add_argument("--style_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--original_colors", type=float, default=1)
    parser.add_argument("--max_nums", type=float, default=1000)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--model", type=str, default='vgg19')
    parser.add_argument("--init_image", type=str, default='image')
    parser.add_argument("--output_image", type=str, default='output.png')
    parser.add_argument("--semantic", type=float, default=1)
    parser.add_argument("--backend", choices=['nn','cudnn'], default='cudnn')
    parser.add_argument("--optimizer", choices=['lbfgs','adam'], default='lbfgs')
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--pooling", choices=['max', 'avg'], default='max')
    parser.add_argument("--save_iter", type=int, default=100)

    args = parser.parse_args()

    start_time = time.time()
    output = run_texture_transfer(args)
    print('run time : {:4f}s'.format(time.time() - start_time))
                 
    save_image(filename=args.output_image, data=output)


if __name__ == "__main__":
    main()


