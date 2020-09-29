import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils
import plotly.graph_objects as go

from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib import cm

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
        
def cal_perturb(model, image, label, vec_x, vec_y, range_x, range_y, 
                 grid_size=50, loss=nn.CrossEntropyLoss(reduction='none'),
                 batch_size=128, device=None):
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
    
    return rx, ry, loss_list, pre_list

def plot_perturb_plotly(rx, ry, loss, predict,
                        z_by_loss=True, color_by_loss=False, color='viridis',
                        min_value=None, max_value=None,
                        title='Loss Visualization', width=600, height=600,
                        x_ratio=1, y_ratio=1, z_ratio=1) :
    
    if z_by_loss :
        zs = loss
    else :
        zs = predict
    
    if color_by_loss :
        colors = loss
    else :
        colors = predict
    
    if min_value is None :
        min_value = int(colors.min())
    if max_value is None :
        max_value = int(colors.max())
        
    fig = go.Figure(data=[go.Surface(z=zs, x=rx, y=ry, surfacecolor=colors, 
                                     colorscale=color, showscale=True, cmin=min_value,
                                     cmax=max_value)],)

#     fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                       project_z=True))

    fig.update_layout(title=title, autosize=True,
                      width=width, height=height, margin=dict(l=65, r=50, b=65, t=90),
                      scene = {
                          "aspectratio": {"x": x_ratio, "y": y_ratio, "z": z_ratio}
                      })
    
    fig.show()
    
def plot_perturb_plt(rx, ry, loss, predict,
                     z_by_loss=True, color_by_loss=False, color='viridis', 
                     min_value=None, max_value=None,
                     title=None, width=8, height=7, linewidth = 0.1,
                     x_ratio=1, y_ratio=1, z_ratio=1,
                     edge_color='#f2fafb', colorbar_yticklabels=None,
                     pane_color=(1.0, 1.0, 1.0, 0.0),
                     tick_pad_x=0, tick_pad_y=0, tick_pad_z=1.5,
                     xticks=None, yticks=None, zticks=None,
                     xlabel=None, ylabel=None, zlabel=None,
                     xlabel_rotation=0, ylabel_rotation=0, zlabel_rotation=0,
                     view_azimuth=230, view_altitude=30,
                     light_azimuth=315, light_altitude=45, light_exag=0) :
    
    if z_by_loss :
        zs = loss
    else :
        zs = predict
    
    if color_by_loss :
        colors = loss
    else :
        colors = predict
    
    xs, ys = np.meshgrid(rx, ry)
    
    fig = plt.figure(figsize=(width, height))
    ax = plt.axes(projection='3d')
    
    if title is not None :
        ax.set_title(title)
    
    if min_value is None :
        min_value = int(colors.min())
    if max_value is None :
        max_value = int(colors.max())
        
    if 'float' in str(colors.dtype):
         scamap = cm.ScalarMappable(cmap=get_cmap(color))
    else:    
         scamap = cm.ScalarMappable(cmap=get_cmap(color, max_value-min_value+1))
   
    scamap.set_array(colors)
    scamap.set_clim(vmax=max_value+.5, vmin=min_value-.5)

    # The azimuth (0-360, degrees clockwise from North) of the light source. Defaults to 315 degrees (from the northwest).
    # The altitude (0-90, degrees up from horizontal) of the light source. Defaults to 45 degrees from horizontal.

    ls = LightSource(azdeg=light_azimuth, altdeg=light_altitude)
    fcolors = ls.shade(colors, cmap=scamap.cmap, vert_exag=light_exag, blend_mode='soft')
    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, facecolors=fcolors,
                           linewidth=linewidth, antialiased=True, shade=False)

    surf.set_edgecolor(edge_color)

    ax.view_init(azim=view_azimuth, elev=view_altitude)

    # You can change 0.01 to adjust the distance between the main image and the colorbar.
    # You can change 0.02 to adjust the width of the colorbar.
    cax = fig.add_axes([ax.get_position().x1+0.01,
                        ax.get_position().y0+ax.get_position().height/8,
                        0.02,
                        ax.get_position().height/4*3])
    
    cbar = plt.colorbar(scamap,
                        ticks=np.linspace(min_value, max_value, max_value-min_value+1),
                        cax=cax)
    
    if colorbar_yticklabels is not None :
        cbar.ax.set_yticklabels(colorbar_yticklabels)
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    
    if xlabel is not None :
        ax.set_xlabel(xlabel, rotation=xlabel_rotation)
    if ylabel is not None :
        ax.set_ylabel(ylabel, rotation=ylabel_rotation)
    if zlabel is not None :
        ax.set_zlabel(zlabel, rotation=zlabel_rotation)
        
    if xticks is not None :
        ax.set_xticks(xticks)
    if yticks is not None :
        ax.set_yticks(yticks)
    if zticks is not None :
        ax.set_zticks(zticks)
    
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    
    ax.tick_params(axis='x', pad=tick_pad_x)
    ax.tick_params(axis='y', pad=tick_pad_y)
    ax.tick_params(axis='z', pad=tick_pad_z)
