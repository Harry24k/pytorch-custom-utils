import math
from collections.abc import Iterable

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torchvision.utils

from ._base import _del_none
    

def _draw_colorbar(ca, cmap=None, colorbar_ticks=None):
    if cmap is not None :
        cbar = plt.colorbar(ca, ticks=np.linspace(0, cmap.N, cmap.N+1))
    else :
        cbar = plt.colorbar(ca)

    if colorbar_ticks is not None : 
        cbar.ax.set_yticklabels(colorbar_ticks)
        

def _to_numpy(input) :
    if isinstance(input, np.ndarray) :
        return input
    else :
        if isinstance(input, torch.Tensor) :
            return input.detach().cpu().numpy()
        else :
            raise RuntimeError("Please input tensor or numpy array.")
        
        
def plot_scatter(ax, input, marker=None, marker_size=None, color=None, label=None,
                 alpha=None, edgecolor=None, linewidth=None, cmap=None, colorbar=False, colorbar_ticks=None) :
    kwargs = {'marker':marker, 's':marker_size, 'c':color, 'label':label, 'alpha':alpha, 
              'cmap':cmap, 'edgecolor':edgecolor, 'linewidth':linewidth}
    _del_none(kwargs)
    
    if cmap is not None :
        kwargs['vmin'] = 0 -.5
        kwargs['vmax'] = (cmap.N-1) + .5
    
    ca = ax.scatter(input[:, 0], input[:, 1], **kwargs)
    
    if colorbar :
        _draw_colorbar(ca, cmap, colorbar_ticks)
    
def plot_line(ax, x, input, linewidth=None, linestyle=None, color=None, label=None, alpha=None) :
    kwargs = {'linewidth':linewidth, 'linestyle':linestyle, 'color':color, 'label':label, 'alpha':alpha}
    _del_none(kwargs)
    ax.plot(x, input, **kwargs)
    
    
def plot_dist(ax, input, kde=False, bins=None, stat=True, norm_hist=False):
    
    input = _to_numpy(input)
    
    if stat :
        print("- Stats")
        print("Max : %f"%np.max(input))
        print("Min : %f"%np.min(input))
        print("Mean : %f"%np.mean(input))
        print("Median : %f"%np.median(input))
    
    sns.distplot(input, kde=kde, bins=bins, norm_hist=norm_hist, ax=ax)
    
    
def plot_img(ax, tensor, ncols=2, normalize=False, range=None, padding=2, pad_value=0) :
    # normalize (bool, optional) – If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False.
    # range (tuple, optional) – tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. By default, min and max are computed from the tensor.
    img = torchvision.utils.make_grid(tensor.detach().cpu(), nrow=ncols, padding=padding,
                                      normalize=normalize, range=range, pad_value=pad_value)
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg,(1,2,0)))
    
            
def _to_array(inputs, lengths) :
    for i, input in enumerate(inputs) :
        if not isinstance(input, Iterable) :
            inputs[i] = [input]*lengths
        
    return inputs

    
def plot_pca(ax, inputs, markers=None, marker_sizes=None,
             colors=None, labels=None, alphas=None,
             cmap=None, colorbar=False, colorbar_ticks=None):
   
    if not isinstance(inputs, list) :
        raise RuntimeError("Please input list.")

    idx = [0]
    for i, input in enumerate(inputs) :
        inputs[i] = _to_numpy(input)
        idx.append(idx[-1]+len(input))
        
    inputs = np.concatenate(inputs)
    inputs = PCA(n_components=2).fit_transform(inputs)
    inputs = [inputs[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
    
    markers, marker_sizes, colors, labels, alphas = _to_array([markers, marker_sizes, colors, labels, alphas], len(inputs))
    for i, input in enumerate(inputs) :
        if colorbar and i == 0:
            ca = plot_scatter(ax, input, marker=markers[i], marker_size=marker_sizes[i],
                         color=colors[i], label=labels[i], alpha=alphas[i], cmap=cmap)
            _draw_colorbar(ca, cmap, colorbar_ticks)
            
        plot_scatter(ax, input, marker=markers[i], marker_size=marker_sizes[i],
                     color=colors[i], label=labels[i], alpha=alphas[i], cmap=cmap)
    
    
def plot_tsne(ax, inputs, markers=None, marker_sizes=None,
              colors=None, labels=None, alphas=None,
              cmap=None, colorbar=False, colorbar_ticks=None):

    if not isinstance(inputs, list) :
        raise RuntimeError("Please input list.")

    idx = [0]
    for i, input in enumerate(inputs) :
        inputs[i] = _to_numpy(input)
        idx.append(idx[-1]+len(input))
        
    inputs = np.concatenate(inputs)
    inputs = TSNE(n_components=2).fit_transform(inputs)
    inputs = [inputs[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
    
    markers, marker_sizes, colors, labels, alphas = _to_array([markers, marker_sizes, colors, labels, alphas], len(inputs))
    for i, input in enumerate(inputs) :
        if colorbar and i == 0:
            ca = plot_scatter(ax, input, marker=markers[i], marker_size=marker_sizes[i],
                         color=colors[i], label=labels[i], alpha=alphas[i], cmap=cmap)
            _draw_colorbar(ca, cmap, colorbar_ticks)
            
        plot_scatter(ax, input, marker=markers[i], marker_size=marker_sizes[i],
                     color=colors[i], label=labels[i], alpha=alphas[i], cmap=cmap)
    
    
#     mulabel = np.array(list(range(len(mu))))
#     cmap = plt.get_cmap('jet', 20)
#     cmap.set_under('gray')

#     plt.scatter(tsnesource[:,0], tsnesource[:,1], marker='x', c=source_label, cmap=cmap,label = 'source', alpha = 0.5)
#     plt.scatter(tsnetarget[:,0], tsnetarget[:,1], marker='o', c=target_label, cmap=cmap,label = 'target', alpha = 0.5)
#     plt.scatter(tsnemu[:,0], tsnemu[:,1], marker="P", c=mulabel, cmap=cmap,label = 'mu')