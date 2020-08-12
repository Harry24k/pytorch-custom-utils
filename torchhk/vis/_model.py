import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils
import plotly.graph_objects as go

from ._base import *
from ._feature import *
from ._feature import _draw_colorbar
    
    
def plot_logit_dist(ax, model, loader, device=None, **kwargs) :
    
    if device is None :
        device = next(model.parameters()).device

    logits = []

    for images, _ in loader :
        images = images.to(device)
        logit = model(images).cpu().detach()
        logits.append(logit.view(-1))        

    input = torch.cat(logits).view(-1).numpy()
    plot_dist(ax, input, **kwargs)
    
    
def plot_grad_dist(ax, model, loader, loss=nn.CrossEntropyLoss(), device=None, **kwargs) :
    
    if device is None :
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
def plot_decision_boundary(ax, model, xrange, yrange, grid_size=500, as_line=False,
                           cmap=None, colorbar=False, colorbar_ticks=None, device=None) :
    
    if device is None :
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
    
    kwargs = {}
    if cmap is not None :
        kwargs['cmap'] = cmap
        kwargs['vmin'] = 0 -.5
        kwargs['vmax'] = (cmap.N-1) + .5
        kwargs['levels'] = np.linspace(kwargs['vmin'], kwargs['vmax'], cmap.N+1)
        
    if as_line :
        ca = ax.contour(x, y, z, **kwargs)
    else :
        ca = ax.contourf(x, y, z, **kwargs)
        
        if colorbar :
            _draw_colorbar(ca, cmap, colorbar_ticks)
        
        
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
        
def plot_perturb(model, image, label, vec_x, vec_y, range_x, range_y, 
                 grid_size=50, color='Viridis', loss=nn.CrossEntropyLoss(reduction='none'),
                 batch_size=128, z_by_loss=True, color_by_loss=True,
                 min_value=0, max_value=9,
                 title='Loss Visualization', width=600, height=600,
                 x_ratio=1, y_ratio=1, z_ratio=1, device=None) :
    
    rx = np.linspace(*range_x, grid_size)
    ry = np.linspace(*range_y, grid_size)
    
    images = []
    loss_list = []
    pre_list = []
    
    if device is None :
        device = next(model.parameters()).device
    
    image = image.to(device)
    label = label.to(device)
    vec_x = vec_x.to(device)
    vec_y = vec_y.to(device)
    
    for j in ry :
        for i in rx :
            images.append(image + i*vec_x + j*vec_y)
            
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
        
    pre_list = np.concatenate(pre_list).reshape(len(rx), len(ry))
    loss_list = np.concatenate(loss_list).reshape(len(rx), len(ry))
    
    if z_by_loss :
        zs = loss_list
    else :
        zs = pre_list
    
    if color_by_loss :
        colors = loss_list
    else :
        colors = pre_list
    
    fig = go.Figure(data=[go.Surface(z=zs, x=rx, y=ry, surfacecolor=colors, 
                                     colorscale=color, showscale=True, cmin=min_value, cmax=max_value)],)
    
#     fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                       project_z=True))
    
    fig.update_layout(title=title, autosize=True,
                      width=width, height=height, margin=dict(l=65, r=50, b=65, t=90),
                      scene = {
                          "aspectratio": {"x": x_ratio, "y": y_ratio, "z": z_ratio}
                      })
    fig.show()