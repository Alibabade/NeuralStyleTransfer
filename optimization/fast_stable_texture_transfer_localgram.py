"""
semantic neural style transfer with content loss and style loss

Input Arguments:
	1. content_mask_image uses white color to label targeted 
           style transfer area
	2. style_mask_image uses black color to label targeted 
           style transfer area.
Notice:
        each mask image only has two colors: white and black

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
        #else:
        #    img = img.resize( (int(math.ceil(size * w / h)), size ), Image.ANTIALIAS)
    if scale is not None:
        
        img = img.resize( (int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    loader = transforms.ToTensor()
                                    
    img = loader(img)
    if not mask: 
        img = img[:3].unsqueeze(0).to(device, torch.float)
    else:
        img = img[:1].unsqueeze(0).to(device, torch.float)
    return img

# preserve original colors of content input
def original_colors(content, output):

    output_y = colors.rgb_to_yuv(output.data)[0][0].unsqueeze(0)
    content_uv = colors.rgb_to_yuv(content.data)[0][1:]#.unsqueeze(0)
    result = torch.cat((output_y, content_uv),0)
    result = colors.yuv_to_rgb(result)
    result = result.unsqueeze(0)
    #result.data.clamp_(0,255.0) #cause a foggy effect 
    return result                    

def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)

def save_image(args, filename, data, content_ori, mask):
    data = deprocess(data)
    data.clamp_(0,1)
    
    if args.semantic == 1:
        content_ori = deprocess(content_ori)
        transpose_mask = 1 - mask
        msk_region = data.mul(mask)
        unmsk_region = content_ori.mul(transpose_mask)
        result = msk_region + unmsk_region
    else:
        result = data
    unloader = transforms.ToPILImage()
    image = result.cpu().clone()
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
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer



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
"""
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a*b, c*d)
    features_t = features.transpose(1, 0)
    G = torch.mm(features, features_t)      
    return G.div(a * b * c *d)

#usually, input dimension is (1 , ch, h, w)
#local gram_matrix: (1, ch, local_ch, h/2, w/2)

def gram_matrix(y):
    
    (b, ch, h, w) = y.size()
    #print(y.size())
    local_ch = 2 #if ch < 64 else 64
    local_wh = w*h
  
    gram = torch.Tensor(ch//local_ch, local_ch, local_ch).to(device)
    for i in range(ch//local_ch):
        features = y[0][i*local_ch:(i+1)*local_ch].view(local_ch, local_wh)
        features_t = features.transpose(1, 0)
        gram[i] = torch.mm(features, features_t)#.div(local_ch*w*h)
    return gram.div(b * local_ch * w * h)
"""
def gram_matrix(y):
    
    (b, ch, h, w) = y.size()
    #print(y.size())
    local_ch = ch // 2 #if ch < 64 else 64
    local_wh = w*h
  
    gram = torch.Tensor(ch, local_ch, local_ch).to(device)
    for i in range(ch):
        indices = [x%ch for x in range(i,i+local_ch)]        
        #features = y[0][i:(i+local_ch)].view(local_ch, local_wh)
        features = y[0,indices].view(local_ch, local_wh)
        features_t = features.transpose(1, 0)
        gram[i] = torch.mm(features, features_t)#.div(local_ch*w*h)
    return gram.div(b * local_ch * w * h)

"""
def gram_matrix(y):
    
    (b, ch, h, w) = y.size()
    #print(y.size())
    local_ch = 4 #compute 4x4 gram matrix for each feature map 
    local_wh = h*w//local_ch
  
    gram = torch.Tensor(ch, local_ch, local_ch).to(device)
    for i in range(ch):
        features = y[0][i].view(local_ch, local_wh)
        features_t = features.transpose(1, 0)
        gram[i] = torch.mm(features, features_t)#.div(w*h)
    return gram.div(b * ch * w * h)
"""

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


#--------------build transform net---------------------------------------------------------------------
def build_transform_model(cnn, args, content_image, style_image, content_mask=None, style_mask=None): 
   
    cnn = copy.deepcopy(cnn)
    
    content_losses = []
    style_losses = []


    model = nn.Sequential()
    out_channels = 0
    if args.tv_weight > 0:
        tv_loss = TVLoss(args.tv_weight)
        model.add_module("tv_loss", tv_loss)
    i = -1 
    for layer in cnn.children():
        
        i += 1        

        if isinstance(layer, nn.Conv2d):            
            name = 'conv_{}'.format(i)
            out_channels = layer.out_channels
            if args.semantic == 1:
                #print('content mask shape',content_mask.shape)
                tmp_net = nn.AvgPool2d(3,1,1).to(device)
                content_mask = tmp_net(content_mask.repeat(1,1,1,1)).detach()
                style_mask = tmp_net(style_mask.repeat(1,1,1,1)).detach()
                #print('content mask shape',content_mask.shape)
            
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
                     style_loss = StyleLoss(target, args.style_weight, 'capture', content_mask, style_mask)
                 elif args.semantic == 0:
                     style_loss = StyleLoss(target, args.style_weight, 'capture')
                 model.add_module("style_layer_{}".format(i), style_loss)
                 style_losses.append(style_loss)
           
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)


            if content_mask is not None:
                content_mask = F.interpolate(content_mask, 
                                size=[int(math.floor(content_mask.shape[2]/2)), 
                                      int(math.floor(content_mask.shape[3]/2))],
                                mode='bilinear',
                                align_corners=False)
            if style_mask is not None:
                style_mask = F.interpolate(style_mask, 
                                size=[int(math.floor(style_mask.shape[2]/2)), 
                                      int(math.floor(style_mask.shape[3]/2))],
                                mode='bilinear',     #bilinears
                                align_corners=False) #False


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
                     style_loss = StyleLoss(target, args.style_weight, 'capture', content_mask, style_mask)
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
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]

    # set mode = 'loss' to style_loss and content_loss modules
    for stl in style_losses:
        stl.mode = 'loss'
    for ctl in content_losses:
        ctl.mode = 'loss'
    if args.tv_weight <= 0:
        tv_loss = None
    return model, content_losses, style_losses, tv_loss
    
        


#----------run texture transfer-------------------------------------------------------------------------------
def run_texture_transfer(args):

    print('running neural style transfer...')
    
    content_image = load_image(args.content_image,mask=False, size=args.image_size, scale=None)
    content_image = preprocess(content_image)
    style_image = load_image(args.style_image,mask=False, size=args.image_size, scale=args.style_scale)
    style_image = preprocess(style_image)

    

    if args.content_mask_image:
        content_mask_image = load_image(args.content_mask_image, mask=True, size=args.image_size, scale=None)  
    else:
        content_mask_image =None
    
    
    if args.style_mask_image:
        style_mask_image = load_image(args.style_mask_image,mask=True, size=args.image_size, scale=args.style_scale)
        transform = transforms.Lambda(lambda x: 1-x)
        style_mask_image = transform(style_mask_image)
    else: 
        style_mask_image = None

    

    

    cnn = None
    if args.model == 'vgg19':
        cnn = models.vgg19(pretrained=True).features.to(device).eval()  
    elif args.model == 'vgg16': 
        cnn = models.vgg16(pretrained=True).features.to(device).eval()
   
    # get tranform net, content losses, style losses
    if args.semantic == 1:
        transform_net, content_losses, style_losses, tv_loss = build_transform_model(cnn, 
                                                                                 args, content_image, 
                                                                                 style_image, 
                                                                                 content_mask_image.clone(), 
                                                                                 style_mask_image.clone())
    else:
        transform_net, content_losses, style_losses, tv_loss = build_transform_model(cnn, 
                                                                                 args, content_image, 
                                                                                 style_image 
                                                                                 )
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
    optimizer = get_input_optimizer(input_img)

    if args.semantic == 1:
        msk = content_mask_image.clone().detach()
        msk = msk.expand_as(input_img)


    print('start optimizing process ...')
    iteration = [0]
    while iteration[0] <= args.max_nums:
        
        def feval():
            #input_img.data.clamp_(0, 256.0)
            optimizer.zero_grad()
            transform_net(input_img)

            

            stloss = 0
            ctloss = 0
            if tv_loss is not None:
                tvloss = tv_loss.loss
            else:
                tvloss = 0
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
                if tv_loss is not None:
                    print("        tv loss        : {:4f}".format(tvloss.item()))
                print("        total loss     : {:4f}".format(loss.item()))

                output_name = args.output_image[:-4] + '_' + str(iteration[0]) + args.output_image[-4:]  
                output = input_img.clone()
                if args.original_colors == 1:
                    output = original_colors(content_image,output)   
                if args.semantic == 1:            
                    save_image(args,filename=output_name, data=output, \
                               content_ori=content_image.clone(), mask=msk.clone())
                else:
                    save_image(args,filename=output_name, data=output, \
                               content_ori=content_image.clone(), mask=None)
                
            return loss
        optimizer.step(feval)
        
    output = input_img.clone()
    if args.original_colors == 1:
        output = original_colors(content_image,output) 
    if args.semantic == 1:
        save_image(args,filename=args.output_image, data=output, content_ori=content_image, mask=msk)
    else:
        save_image(args,filename=args.output_image, data=output, content_ori=content_image, mask=None)
    
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
    parser.add_argument("--content_layers", type=list, default=['23'])  #4,9,16,23
    parser.add_argument("--style_layers", type=list,
                        default=['2','7','12','21','30'])
    parser.add_argument("--hist_layers", type=list, 
                        default=['2','21'])
    # options for weights
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument("--style_weight", type=float, default=1e6)
    parser.add_argument("--hist_weight", type=float, default=1)
    parser.add_argument("--tv_weight", type=float, default=1e-3)
    # alternative options 
    parser.add_argument("--style_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--original_colors", type=float, default=1)
    parser.add_argument("--max_nums", type=float, default=100)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--model", type=str, default='vgg19')
    parser.add_argument("--init_image", type=str, default='image')
    parser.add_argument("--output_image", type=str, default='output.png')
    parser.add_argument("--semantic", type=float, default=1,
                        help="0 denotes no mask; 1 denotes masks for semantic neural style transfer.")
    parser.add_argument("--backend", choices=['nn','cudnn'], default='cudnn')

    args = parser.parse_args()

    start_time = time.time()
    run_texture_transfer(args)
    print('run time : {:4f}s'.format(time.time() - start_time))
                 


if __name__ == "__main__":
    main()


