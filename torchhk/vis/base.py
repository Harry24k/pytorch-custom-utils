import matplotlib.pyplot as plt
import torch

# TODO : Palette
palette = {
    
}

        
#              colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
#  (1.0, 0.4980392156862745, 0.054901960784313725),
#  (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
#  (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
#  (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
#  (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
#  (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
#  (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
#  (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
#  (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)],
#         tableau20 = [[ 31, 119, 180],
#              [255, 127,  14],
#              [ 44, 160,  44],
#              [214,  39,  40],
#              [148, 103, 189],
#              [140,  86,  75],
#              [227, 119, 194],
#              [127, 127, 127],
#              [188, 189,  34],
#              [ 23, 190, 207]]

#         for i in range(len(tableau20)):  
#             r, g, b = tableau20[i]  
#             tableau20[i] = (r / 255., g / 255., b / 255.)  


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
    return mean - range/2 * (1+ratio), mean + range/2 * (1+ratio)


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