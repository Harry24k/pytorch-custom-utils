import matplotlib.pyplot as plt
import torch

# TODO : Palette
palette = {
    
}

# TODO : Params
# import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'} #'font.size': 22
# pylab.rcParams.update(params)

            
def _add_margin(x_min, x_max, ratio=0.3):
    range = x_max - x_min
    mean = (x_max + x_min) / 2
    return mean + range/2 * (1+ratio), mean - range/2 * (1+ratio)


def init_plot(ax=None, figsize=(6,6), title="", xlabel="", ylabel="",
              xlim=None, ylim=None, pad_ratio=0, show_axis=True, tight=True) :
    
    if ax is None :
        ax = plt.subplots(1, 1, figsize=figsize)[1]
        
    if xlim is not None :
        xlim = _add_margin(*xlim, pad_ratio)
        ax.set_xlim(xlim)
        
    if ylim is not None :
        ylim = _add_margin(*ylim, pad_ratio)
        ax.set_ylim(ylim)
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
                
    if not show_axis :
        ax.axis('off')
    
    if tight :
        plt.tight_layout()
    
    return ax