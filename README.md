# pytorch-custom-utils

This is a custom useful utils for Pytorch.

## Usage

### Dependencies

- torch 1.4.0
- torchvision 0.5.0
- python 3.6
- matplotlib 2.2.2
- numpy 1.14.3
- seaborn 0.9.0
- copy
- warnings

### Installation

- `pip install torchhk` or
- `git clone https://github.com/Harry24k/pytorch-custom-utils`

```python
from torchhk import *
```

### Demos
* **rm(RecordManager)** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/RecordManager.ipynb)): 
RecordManager will help you to watch records pretty during iterations. It also provides some useful functions such as summary, plot, etc.

* **datasets(Datasets)** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Datasets.ipynb)): 
Datasets will help you to bring torchvision data with only simple one line. It also has split function for people who need a validation set. 

* **vis(Vis)** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Vis.ipynb)): 
Vis provides two functions as below :
    * imshow : It will help you to see torch tensor images in one plot.
    * weightshow : It will help you to see all weights' distribution of a torch model.

* **transform(Transform)** ([code](https://github.com/Harry24k/pytorch-custom-utils/blob/master/demo/Transform.ipynb)): 
Transform will help you to construct a new model with certain layers changed.

## Update Records

### Version 0.1
* **RecordManager**
* **Datasets**
* **Vis** : 
    * imshow
    * weightshow
* **Transform** :
    * transform_layer
    * transform_model

### Version 0.2
* **RecordManager**
    * Ploting two axis added.
* **Datasets**
    * CIFAR100 added.
* **Vis** : 
    * imshow -> im_show 
    * weightshow -> weight_show 
* **Transform** :
    * transform_layer
    * transform_model

### Version 0.3
* **RecordManager**
* **Transform** :
    * transform_model : Error Solved. 

### Version 0.4
* **Datasets** :
    * Shuffle set to False in validation set.