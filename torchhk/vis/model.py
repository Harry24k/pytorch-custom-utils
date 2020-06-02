import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils
import plotly.graph_objects as go

from .base import *
from .feature import *
    
    
def plot_logit_dist(ax, model, loader, **kwargs) :
    
    device = next(model.parameters()).device

    logits = []

    for images, _ in loader :
        images = images.to(device)
        logit = model(images).cpu().detach()
        logits.append(logit.view(-1))        

    input = torch.cat(logits).view(-1).numpy()
    plot_dist(ax, input, **kwargs)
    
    
def plot_grad_dist(ax, model, loader, loss=nn.CrossEntropyLoss(), **kwargs) :
    
    device = next(model.parameters()).device

    grads = []
    
    for images, labels in loader :
        images = images.to(device)
        labels = labels.to(device)
        
        images.requires_grad = True
        outputs = model(images)
        cost = loss(outputs, labels).to(device)
        
        grad = torch.autograd.grad(cost, images, 
                                   retain_graph=False, create_graph=False)[0]
        grad = grad.cpu().detach()
        grads.append(grad.view(-1))        

    input = torch.cat(grads).view(-1).numpy()
    plot_dist(ax, input, **kwargs)
    

# TODO : Colorbar
def plot_decision_boundary(ax, model, xrange, yrange, grid_size=500, as_line=False, colorbar=False, levels=None) :
    
    device = next(model.parameters()).device
    
    # Create Meshgrid
    x_0 = torch.linspace(*xrange, grid_size)
    y_0 = torch.linspace(*yrange, grid_size)
    xv, yv = torch.meshgrid(x_0, y_0)
    
    # Get Predicted Value
    xv = xv.reshape(-1, 1)
    yv = yv.reshape(-1, 1)
    z = model(torch.cat([xv, yv], dim=1).to(device))

    # Reshape to grid_size*grid_size
    x = xv.detach().cpu().numpy().reshape(grid_size, grid_size)
    y = yv.detach().cpu().numpy().reshape(grid_size, grid_size)
    z = z.detach().cpu().numpy().reshape(grid_size, grid_size)
    
    if as_line :
        ax.contour(x, y, z, levels=levels)
    else :
        ax.contourf(x, y, z, levels=levels)
        
        
def plot_weight(ax, model, filter, **kwargs):
    params = []

    for name, param in model.named_parameters():
        if param.requires_grad and filter(name) :
            params.append(param.view(-1).cpu().detach())        

    input = torch.cat(params).view(-1).numpy()
    plot_dist(ax, input, **kwargs)
    
    
def plot_individual_weight(model, ncols=2, figsize=(5,5), 
                           title=None, filter=lambda x:True,
                           xlabel='$q_{ij}$', ylabel='Counts',
                           xlim=None, ylim=None, **kwargs) :

    #TODO : Subplot of Subplot for given ax.
    
    subplots_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad and filter(name) :
            subplots_num += 1

    nrows = math.ceil(subplots_num/ncols)
    xsize, ysize = figsize
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols*xsize, nrows*ysize))

    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad and filter(name) :
            if ncols == 1 :
                ax = axes[i//ncols]
            elif len(axes.shape) == 1 :
                ax = axes[i]
            else :
                ax = axes[i//ncols, i%ncols]
            data = param.view(-1).cpu().detach().numpy()
            sns.distplot(data, ax=ax, **kwargs)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)    

            ax.set_title(name)
            i += 1
        
def plot_perturb(model, image, label, loss, d1, d2, r1, r2,
                 batch_size=128, z_by_loss=True, color_by_loss=True,
                 title='Loss Visualization', width=600, height=600,
                 x_ratio=1, y_ratio=1, z_ratio=1) :
    
    images = []
    loss_list = []
    pre_list = []
    
    device = next(model.parameters()).device
    
    image = image.to(device)
    label = label.to(device)
    d1 = d1.to(device)
    d2 = d2.to(device)
    
    for i in r1 :
        for j in r2 :
            images.append(image + i*d1 + j*d2)
            
            if len(images) == batch_size :
                images = torch.stack(images)
                labels = torch.stack([label]*batch_size)
                outputs = model(images)
                
                _, pres = torch.max(outputs.data, 1)
                pre_list.append(pres.data.cpu().numpy())
                loss_list.append(loss(outputs, labels).data.cpu().numpy())
                    
                images = []

    images = torch.stack(images)
    labels = torch.stack([label]*len(images))
    outputs = model(images)

    _, pres = torch.max(outputs.data, 1)
    pre_list.append(pres.data.cpu().numpy())
    loss_list.append(loss(outputs, labels).data.cpu().numpy())
        
    pre_list = np.concatenate(pre_list).reshape(len(r1), len(r2))
    loss_list = np.concatenate(loss_list).reshape(len(r1), len(r2))
    
    if z_by_loss :
        zs = loss_list
    else :
        zs = pre_list
    
    if color_by_loss :
        colors = loss_list
    else :
        colors = pre_list
    
    fig = go.Figure(data=[go.Surface(z=zs, x=r1, y=r2, surfacecolor=colors,
                                     colorscale='Viridis', showscale=True)],)
    
#     fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                       project_z=True))
    
    fig.update_layout(title=title, autosize=True,
                      width=width, height=height, margin=dict(l=65, r=50, b=65, t=90),
                      scene = {
                          "aspectratio": {"x": x_ratio, "y": y_ratio, "z": z_ratio}
                      })
    fig.show()