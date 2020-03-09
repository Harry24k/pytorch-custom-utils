import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils

def weight_show(model, ncols=2, figsize=(5,5),
                xlabel='$q_{ij}$', ylabel='Counts', save_path=None) :
    
    if not isinstance(model, nn.Module) :
        raise ValueError(reduction + " is not valid")

    xsize, ysize = figsize
    
    subplots_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            subplots_num += 1

    nrows = math.ceil(subplots_num/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols*xsize, nrows*ysize))

    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ncols == 1:
                ax = axes[i//ncols]
            else :
                ax = axes[i//ncols, i%ncols]
            data = param.view(-1).cpu().detach().numpy()
            sns.distplot(data, kde=False, ax=ax)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(name)
            i += 1

    plt.tight_layout()
    if save_path is not None :
        plt.savefig(save_path)
        print("Figure Saved!")
    plt.show()
    plt.clf()
    
def im_show(tensor, title="", figsize=(5,15), ncols=8, normalize=False, range=None,
           scale_each=False, padding=2, pad_value=0, save_path=None) :
    
    # tensor (Tensor or list) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
    # nrow (python:int, optional) – Number of images displayed in each row of the grid. The final grid size is (B / nrow, nrow). Default: 8.
    # padding (python:int, optional) – amount of padding. Default: 2.
    # normalize (bool, optional) – If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False.
    # range (tuple, optional) – tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. By default, min and max are computed from the tensor.
    # scale_each (bool, optional) – If True, scale each image in the batch of images separately rather than the (min, max) over all images. Default: False.
    # pad_value (python:float, optional) – Value for the padded pixels. Default: 0.

    img = torchvision.utils.make_grid(tensor, ncols, padding, normalize, range, scale_each, pad_value)
    npimg = img.numpy()
    fig = plt.figure(figsize = figsize)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    if save_path is not None :
        plt.savefig(save_path)
    
    plt.clf()