"""
neural style transfer with content loss and style loss

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
import torchvision.models as models
import pytorch_colors as colors
import copy 
import argparse
import operator
from torch.utils.data import DataLoader
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



# utilize the LBFGS optimizer
def get_input_optimizer(input_img, args, model):
    # this line to show that input is a parameter that requires a gradient
    #optimizer = optim.LBFGS([input_img.requires_grad_()])
    #return optimizer
    if args.optimizer == 'lbfgs':
        print('Using L-BFGS optimizer...')
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif args.optimizer == 'adam':
        print('Using Adam optimizer...')
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    return optimizer


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

#for batch_size > 1
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a, b, c*d)
    features_t = features.transpose(1, 2)
    G = features.bmm(features_t)  # compute the gram product
    return G.div(b * c *d)


class ContentLoss(nn.Module):
    def __init__(self, weight, target_feature=None, mode=None):
        super(ContentLoss, self).__init__()
        self.target = torch.Tensor()
        self.weight = weight        
        self.mode = mode
        self.loss = 0       
    def forward(self, input):
        self.input = input
        if self.mode == 'capture':
             self.target = input.detach()
        
        if self.mode == 'loss':            
             self.loss = F.mse_loss(input, self.target) * self.weight
        return input

class StyleLoss(nn.Module):

    def __init__(self, weight, target_feature=None,  mode=None):
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
            self.G = gram_matrix(input)
            self.loss = F.mse_loss(self.G, self.target) * self.weight

        return input


#--------------build transform net---------------------------------------------------------------------
def build_loss_model(cnn, args,  style_image): 
   
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
            
            if str(i) in args.content_layers:
                 for stl in style_losses:
                     stl.mode = None
                 print ('setting up content layer...')
                 content_loss = ContentLoss(args.content_weight)
                 model.add_module("content_layer_{}".format(i), content_loss)
                 content_losses.append(content_loss)

            if str(i) in args.style_layers:
                 for ctl in content_losses:
                     ctl.mode = None
                 print ('capturing style target...')
                 target = model(style_image).detach()               
                 style_loss = StyleLoss(args.style_weight, target, 'capture')
                 model.add_module("style_layer_{}".format(i), style_loss)
                 style_losses.append(style_loss)
           
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            
            if str(i) in args.content_layers:
                 for stl in style_losses:
                     stl.mode = None
                 print ('setting up content layer...')
                 
                 content_loss = ContentLoss(args.content_weight)
                 model.add_module("content_layer_{}".format(i), content_loss)
                 content_losses.append(content_loss)

            if str(i) in args.style_layers:
                 for ctl in content_losses:
                     ctl.mode = None
                 print ('capturing style target...')
                 target = model(style_image).detach()
                 style_loss = StyleLoss(args.style_weight, target, 'capture')
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
    #print(model)

    # set mode = 'loss' to style_loss and content_loss modules
    for stl in style_losses:
        stl.mode = 'None'
    for ctl in content_losses:
        ctl.mode = 'None'
    if args.tv_weight <= 0:
        tv_loss = None
    return model, content_losses, style_losses, tv_loss
    
        
#---------------------------------------training stage-----------------------------------------

def run_train(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('running training processing...')
    

    style_image = load_image(args.style_image,mask=False, 
                             size=args.image_style_size, scale=args.style_scale,
                             square=True)
    style_image = preprocess(style_image)

    #save_image('style_image.png',style_image)
    
    cnn = None
    if args.loss_model == 'vgg19':
        cnn = models.vgg19(pretrained=True).features.to(device).eval()  
    elif args.loss_model == 'vgg16': 
        cnn = models.vgg16(pretrained=True).features.to(device).eval()


    # get tranform net, content losses, style losses
    loss_net, content_losses, style_losses, tv_loss = build_loss_model(cnn, 
                                                                       args, 
                                                                       style_image)
    #print(loss_net)
    

    #collect space back 
    cnn = None
    del cnn

    if args.backend == 'cudnn':
       torch.backends.cudnn.enabled=True
   
    #this is to define the inchannels of transferm_net
    in_channels = 3


    transform = transforms.Compose([
                           transforms.Resize(args.image_size),     
                           transforms.CenterCrop(args.image_size),                      
                           transforms.ToTensor(),   
                           ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transform_net = TransformerNet(in_channels).to(device)
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

            #save_image('content_x1.png', x[3].unsqueeze(0))
            #assert 0 == 1

            #forward input to transform_net
            y = transform_net(x)

            #compute pixel loss          
            pixloss = 0. 
            if args.pixel_weight > 0:
               pixloss = mse_loss(x, y) * args.pixel_weight

            #compute content loss and style loss
            
            for ctl in content_losses:
                ctl.mode = 'capture'
            for stl in style_losses:
                stl.mode = 'None'
            loss_net(x_ori)

            for ctl in content_losses:
                ctl.mode = 'loss'
            for stl in style_losses:
                stl.mode = 'loss'
            loss_net(y)
            for ctl in content_losses:
                ctloss += mse_loss(ctl.target, ctl.input) * args.content_weight           
            for stl in style_losses:        
                local_G = gram_matrix(stl.input)
                stloss +=  mse_loss(local_G, stl.target) * args.style_weight                
            if tv_loss is not None:
                tvloss = tv_loss.loss
            else:
                tvloss = 0.

            loss = ctloss + stloss + pixloss #+ tvloss
            
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
                                      batch_id + 1) +  ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transform_net.state_dict(), ckpt_model_path)
                transform_net.to(device).train()

               
        iteration[0] += 1
                
           
    #save final model
    transform_net.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(
                          ' ', '_') + "_content_" + str(args.content_weight) + "_style_" + str(
                          args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transform_net.state_dict(), save_model_path)

    print("\n training process is Done!, trained model saved at", save_model_path)    


#----------run feedward texture transfer-----------------------------------------------------------------------

def run_feedforward_texture_transfer(args):

    print('running feedforward neural style transfer...')
    
    content_image = load_image(args.content_image,mask=False, 
                               size=args.image_size, scale=None,
                               square=False)
    content_image = preprocess(content_image)


    input_image = content_image
  

    in_channels = 3

    stylizing_net = TransformerNet(in_channels)
    state_dict = torch.load(args.style_model)

    for k in list(state_dict.keys()):
        if re.search(r'in\d+\/running_(mean|var)$', k):
            del state_dict[k]
    del k
    stylizing_net.load_state_dict(state_dict)
    stylizing_net = stylizing_net.to(device)

    output = stylizing_net(input_image)
    
    if args.original_colors == 1:
        output = original_colors(content_image.cpu(),output)    
    save_image(filename=args.output_image, data=output.detach()) 


    
#---------------add arguments for inputs-------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description='parser for feedforward neural normal transfer')
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)

    # options for layers
    parser.add_argument("--content_layers", type=list, default=['9'])
    parser.add_argument("--style_layers", type=list,
                        default=['4','9','16','23'])
                        #default=['4','9','16'])


    # options for weights
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument("--style_weight", type=float, default=1e2)
    parser.add_argument("--tv_weight", type=float, default=1e-6)
    parser.add_argument("--pixel_weight", type=float, default=0)
    
    # options for images
    #parser.add_argument("--content_image", type=str, required=True,
    #                    help="please add the path to your conent image.")
    parser.add_argument("--style_image", type=str, required=False,
                        help="please add the path to your style image.")
    parser.add_argument("--content_mask_image", type=str, required=False,
                        help="please add the path to your content mask image.")
    parser.add_argument("--style_mask_image", type=str, required=False,
                        help="please add the path to your style mask image.")
      
    # alternative options 
    parser.add_argument("--style_scale", type=float, default=1)
    parser.add_argument("--image_style_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--original_colors", type=float, default=0)
    parser.add_argument("--gpu", type=float, default=0) 
    parser.add_argument("--backend", choices=['nn','cudnn'], default='cudnn')
    parser.add_argument("--optimizer", choices=['lbfgs','adam'], default='adam')
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--pooling", choices=['max', 'avg'], default='max')
    parser.add_argument("--loss_model", type=str, default='vgg16')

    #options for stylization
    parser.add_argument("--content_image", type=str, required=False,
                        help="please add the path to your conent image.")
    parser.add_argument("--output_image", type=str, default='output.png')
    parser.add_argument("--style_model", type=str, default='models/checkpoints/mix/ckpt_epoch_1_batch_id_2000.pth')
    

    args = parser.parse_args()

    start_time = time.time()
    if args.training == 1:
        run_train(args)
        print('run time : {:4f} h'.format((time.time() - start_time)/3600))
    else:
        run_feedforward_texture_transfer(args)
        print('run time : {:4f}s'.format(time.time() - start_time))
                 
    #


if __name__ == "__main__":
    main()


