import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors
import seaborn as sns
import torch

def init_settings(font_size=None, title_size=None, label_size=None,
                  xlabel_size=None, ylabel_size=None, legend_size=None) :
    
    params = {'font.size': font_size,
              'axes.titlesize': title_size,
              'axes.labelsize': label_size,
              'xtick.labelsize': xlabel_size,
              'ytick.labelsize': ylabel_size,
              'legend.fontsize': legend_size,}
    _del_none(params)
    pylab.rcParams.update(params)
    print("rcParams updated.")

def get_cmap(input=None, num=-1) :
    c = sns.color_palette(input)
    if num != -1 :
        if input is None :
            raise RuntimeError("Can't generate a cmap without any inputs")
        else :
            c = c[:num]
    return matplotlib.colors.ListedColormap(c)

def init_plot(ax=None, figsize=(6,6), title=None, xlabel=None, ylabel=None,
              xlim=None, ylim=None, pad_ratio=0, show_axis=True, show_grid=False, tight=True) :
    
    if ax is None :
        ax = plt.subplots(1, 1, figsize=figsize)[1]
        
    if title is not None :
        ax.set_title(title)
        
    if xlabel is not None :
        ax.set_xlabel(xlabel)
    
    if ylabel is not None :
        ax.set_ylabel(ylabel)
        
    if xlim is not None :
        xlim = _add_margin(*xlim, pad_ratio)
        ax.set_xlim(xlim)
        
    if ylim is not None :
        ylim = _add_margin(*ylim, pad_ratio)
        ax.set_ylim(ylim)
                
    if not show_axis :
        ax.axis('off')
        
    if show_grid :
        ax.grid()
    
    if tight :
        plt.tight_layout()
    
    return ax


def make_twin(ax, ylabel=None, ylim=None, pad_ratio=0) :
    
    ax2 = ax.twinx()
    
    if ylabel is not None :
        ax2.set_ylabel(ylabel)
        
    if ylim is not None :
        ylim = _add_margin(*ylim, pad_ratio)
        ax2.set_ylim(ylim)
    
    return ax2
            
def _add_margin(x_min, x_max, ratio=0.3):
    range = x_max - x_min
    mean = (x_max + x_min) / 2
    return mean - range/2 * (1+ratio), mean + range/2 * (1+ratio)

def _del_none(input) :
    for key in input.keys() :
        if input is None :
            del input[key]