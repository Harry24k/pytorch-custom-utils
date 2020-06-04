# pytorch-custom-utils

Custom utils for Pytorch.

## Usage

### Dependencies

- torch 1.4.0
- torchvision 0.5.0
- python 3.6
- matplotlib 2.2.2
- numpy 1.14.3
- seaborn 0.9.0
- sklearn
- plotly

### Installation

- `pip install torchhk` or
- `git clone https://github.com/Harry24k/pytorch-custom-utils`

```python
from torchhk import *
```

### Demos
* **RecordManager** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/RecordManager.ipynb), [markdown](https://github.com/Harry24k/pytorch-custom-utils/blob/master/docs/RecordManager.md)): 
RecordManager will help you to record loss or accuracy during iterations. It also provides some useful functions such as summary, plot, etc.

* **Datasets** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Datasets.ipynb), [markdown](https://github.com/Harry24k/pytorch-custom-utils/blob/master/docs/Datasets.md)): Dataset utils for loading, split and label filtering.
    > Supported Datasets: CIFAR10, CIFAR100, STL10, MNIST, FashionMNIST, SVHN, MNISTM, ImageNet, USPS.

* **Vis** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Vis.ipynb), [markdown](https://github.com/Harry24k/pytorch-custom-utils/blob/master/docs/Vis.md)): Visualization tools for torch tensors.

* **Transform** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Transform.ipynb)): 
Transform will help you to construct a new model with certain layers changed.