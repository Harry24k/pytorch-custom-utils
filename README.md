# pytorch-custom-utils

[![License](https://img.shields.io/github/license/Harry24k/pytorch-custom-utils)](https://img.shields.io/github/license/Harry24k/pytorch-custom-utils)
[![Pypi](https://img.shields.io/pypi/v/torchhk.svg)](https://img.shields.io/pypi/v/torchhk)

This is a lightweight repository to help PyTorch users.

## Usage

### :clipboard: Dependencies

- torch 1.4.0
- torchvision 0.5.0
- python 3.6
- matplotlib 2.2.2
- numpy 1.14.3
- seaborn 0.9.0
- sklearn
- plotly

### :hammer: Installation

- `pip install torchhk` or
- `git clone https://github.com/Harry24k/pytorch-custom-utils`

```python
from torchhk import *
```

### :rocket: Demos

- **RecordManager** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/RecordManager.ipynb), [markdown](https://github.com/Harry24k/pytorch-custom-utils/blob/master/docs/RecordManager.md)): 
RecordManager will help you to keep tracking training records.

- **Datasets** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Datasets.ipynb), [markdown](https://github.com/Harry24k/pytorch-custom-utils/blob/master/docs/Datasets.md)): 
Dataset will help you to use torch datasets including split and label-filtering.

<details><summary>Supported datasets</summary><p>

```python
# CIFAR10
datasets = Datasets("CIFAR10", root='./data')
    
# CIFAR100
datasets = Datasets("CIFAR100", root='./data')
    
# STL10
datasets = Datasets("STL10", root='./data')
    
# MNIST
datasets = Datasets("MNIST", root='./data')
    
# FashionMNIST
datasets = Datasets("FashionMNIST", root='./data')
    
# SVHN
datasets = Datasets("SVHN", root='./data')
    
# MNISTM
datasets = Datasets("MNISTM", root='./data')
    
# ImageNet
datasets = Datasets("ImageNet", root='./data')
    
# USPS
datasets = Datasets("USPS", root='./data')
    
# TinyImageNet
datasets = Datasets("TinyImageNet", root='./data')
    
# CIFAR with Unsupervised
datasets = Datasets("CIFAR10U", root='./data')
datasets = Datasets("CIFAR100U", root='./data')
    
# Corrupted CIFAR (Only test data will be corrupted)
# CORRUPTIONS = [
#    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
#    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
#    'brightness', 'contrast', 'elastic_transform', 'pixelate',
#    'jpeg_compression'
#]
datasets = Datasets("CIFAR10", root='./data',corruption='gaussian_noise')
```
</p></details>

- **Vis** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Vis.ipynb), [markdown](https://github.com/Harry24k/pytorch-custom-utils/blob/master/docs/Vis.md)): 
Vis will help you to visualize torch tensors.

- **Transform** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Transform.ipynb)): 
Transform will help you to change specific layers.


## Contribution

Contribution is always welcome! Use [pull requests](https://github.com/Harry24k/adversarial-attacks-pytorch/pulls) :blush:
