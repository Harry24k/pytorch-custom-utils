
# Vis

<!-- MarkdownTOC autolink="true" lowercase="false" -->

- [Base](#Base)
    - [Create Plot](#Create-Plot)
    - [Subplots](#Subplots)
    - [Twinplot](#Twinplot)
    - [Cmap](#Cmap)
- [Feature](#Feature)
    - [plot_scatter](#plot_scatter)
    - [plot_dist](#plot_dist)
    - [plot_img](#plot_img)
    - [plot_pca](#plot_pca)
    - [plot_tsne](#plot_tsne)
- [Model](#Model)
    - [plot_logit_dist](#plot_logit_dist)
    - [plot_grad_dist](#plot_grad_dist)
    - [plot_decision_boundary](#plot_decision_boundary)
    - [plot_weight](#plot_weight)
    - [plot_individual_weight](#plot_individual_weight)
    - [plot_perturb](#plot_perturb)

<!-- /MarkdownTOC -->

```python
from torchhk.vis import *
```

## Base

### Create Plot


```python
ax = init_plot(ax=None, figsize=(3,3), title="", xlabel="", ylabel="",
               xlim=None, ylim=None, pad_ratio=0, show_axis=True, tight=True)
```


![png](output_4_0.png)


### Subplots


```python
fig, ax = plt.subplots(1, 3, figsize=(9,3))
ax1 = init_plot(ax=ax[0], title="Figure 1")
ax2 = init_plot(ax=ax[1], title="Figure 2")
ax3 = init_plot(ax=ax[2], title="Figure 3")
```


![png](output_6_0.png)


### Twinplot


```python
ax1 = init_plot(title="Figure 1", ylabel="First")
ax2 = make_twin(ax1, ylabel="Second")
```


![png](output_8_0.png)


### Cmap


```python
cmap = get_cmap(input='tab10', num=10)
sns.palplot(cmap.colors)
```


![png](output_10_0.png)



```python
cmap = get_cmap(input='tab10', num=10)
sns.palplot(cmap.colors)
```


![png](output_11_0.png)



```python
cmap = get_cmap(input=["#9b59b6", "#3498db", "#95a5a6"])
sns.palplot(cmap.colors)
```


![png](output_12_0.png)


## Feature

### plot_scatter


```python
ax = init_plot(figsize=(3,3), title="1 Scatter")
plot_scatter(ax, torch.rand(100, 2))
```


![png](output_15_0.png)



```python
ax = init_plot(figsize=(3,3), title="2 Scatter")
plot_scatter(ax, torch.rand(100, 2), color='red', marker='o', marker_size=10)
plot_scatter(ax, torch.rand(100, 2), color='blue', marker='x', marker_size=20)
```


![png](output_16_0.png)



```python
ax = init_plot(figsize=(3,3), title="Scatter with Colorbar")
plot_scatter(ax, torch.rand(100, 2), color=torch.randint(low=0, high=3, size=torch.rand(100).shape),
             marker='o', marker_size=10, cmap=cmap, colorbar=True, colorbar_ticks=['a', 'b', 'c'])
```


![png](output_17_0.png)


### plot_dist


```python
ax = init_plot(figsize=(3,3), title="2 Scatter")
plot_dist(ax, torch.rand(100), bins=[0, 0.2, 0.4, 0.6, 0.8, 1], stat=True, norm_hist=False)
```

    - Stats
    Max : 0.999406
    Min : 0.014376
    Mean : 0.520582
    Median : 0.523628
    


![png](output_19_1.png)


### plot_img


```python
from torchhk.datasets import *
```


```python
data =Datasets(data_name="CIFAR10")
train_loader, test_loader = data.get_loader(batch_size=12)
train_images, _ = iter(train_loader).next()
```

    Files already downloaded and verified
    Files already downloaded and verified
    Data Loaded!
    Train Data Length : 50000
    Test Data Length : 10000
    


```python
ax = init_plot(figsize=(5, 5), title="Tensor Images")
plot_img(ax, train_images, ncols=3, padding=3, pad_value=0.5)
```


![png](output_23_0.png)


### plot_pca


```python
ax = init_plot(figsize=(5, 5), title="PCA")
plot_pca(ax, [torch.rand(100, 3), torch.rand(100, 3)], colors=['blue', 'red'])
```


![png](output_25_0.png)


### plot_tsne


```python
ax = init_plot(figsize=(5, 5), title="TSNE")
plot_pca(ax, [torch.rand(100, 3), torch.rand(100, 3)], colors=['blue', 'red'], alphas=[0.2, 0.5])
```


![png](output_27_0.png)


## Model


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
```


```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1,32,5),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(32,64,5),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*4*4, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.ReLU(),
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )       
        
    def forward(self,x):
        out = self.conv_layer(x)
        out = self.fc_layer(out)

        return out
```


```python
model = CNN().cuda()
```


```python
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
num_epochs = 2
```


```python
data = Datasets(data_name="MNIST")
train_loader, test_loader = data.get_loader(batch_size=128)
```

    Data Loaded!
    Train Data Length : 60000
    Test Data Length : 10000
    


```python
for epoch in range(num_epochs):
    
    for i, (batch_images, batch_labels) in enumerate(train_loader):
        
        X = batch_images.cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i+1) % 200 == 0:
            print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_loader), cost.item()))
```

    Epoch [1/2], lter [200/468], Loss: 2.3004
    Epoch [1/2], lter [400/468], Loss: 2.3018
    Epoch [2/2], lter [200/468], Loss: 2.3126
    Epoch [2/2], lter [400/468], Loss: 2.3093
    


```python
correct = 0
total = 0

for images, labels in test_loader:
    
    images = images.cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Accuracy of test images: %f %%' % (100 * float(correct) / total))
```

    Accuracy of test images: 11.350000 %
    

### plot_logit_dist


```python
ax = init_plot(figsize=(5, 5), title="Logit DIstributions")
plot_logit_dist(ax, model, train_loader)
```

    - Stats
    Max : 0.161103
    Min : -0.232128
    Mean : 0.002596
    Median : -0.000358
    


![png](output_37_1.png)


### plot_grad_dist


```python
ax = init_plot(figsize=(5, 5), title=r"Gradient Distributions ($\nabla_x L$)")
plot_grad_dist(ax, model, train_loader, loss=nn.CrossEntropyLoss(), bins=[-0.1, -0.05, 0, 0.05, 0.1])
```

    - Stats
    Max : 0.000000
    Min : 0.000000
    Mean : 0.000000
    Median : 0.000000
    


![png](output_39_1.png)


### plot_decision_boundary


```python
class latent() :
    def __init__(self, model) :
        self.model = model
        
    def parameters(self) :
        return self.model.fc_layer[4:].parameters()
    
    def __call__(self, x) :
        out = self.model.fc_layer[4:](x)
        _, predicted = torch.max(out, 1)
        return predicted
        
latent_model = latent(model)
```


```python
ax = init_plot(figsize=(5, 5), title="Decision Boundary")
plot_decision_boundary(ax, latent_model, xrange=(-10, 10), yrange=(-10, 10))
```


![png](output_42_0.png)



```python
cmap = get_cmap('tab10', 10)

ax = init_plot(figsize=(5, 5), title="Decision Boundary with Colorbar")
plot_decision_boundary(ax, latent_model, xrange=(-10, 10), yrange=(-10, 10),
                       cmap=cmap, colorbar=True)#, colorbar_ticks=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
```


![png](output_43_0.png)



```python
cmap = get_cmap(['#000000']*10)
ax = init_plot(figsize=(5, 5), title="Decision Boundary without fill")
plot_decision_boundary(ax, latent_model, xrange=(-10, 10), yrange=(-10, 10), as_line=True, cmap=cmap)
```


![png](output_44_0.png)


### plot_weight


```python
def filter(name) :
    isWeight = "weight" in name
    return isWeight
```


```python
ax = init_plot(figsize=(5, 5), title="Weight")
plot_weight(ax, model, filter=filter)
```

    - Stats
    Max : 0.438365
    Min : -0.428861
    Mean : -0.000034
    Median : -0.000063
    


![png](output_47_1.png)


### plot_individual_weight


```python
def filter(name) :
    isWeight = "weight" in name
    isConv = "conv" in name
    return isWeight*isConv
```


```python
plot_individual_weight(model, ncols=1, filter=filter)
```


![png](output_50_0.png)


### plot_perturb


```python
test_images, test_labels = iter(test_loader).next()
```


```python
plot_perturb(model=model, image=test_images[0], label=test_labels[0],
             vec_x=torch.rand_like(test_images[0]), vec_y=2/255*torch.randint(-1, 1, test_images[0].shape),
             range_x=(-2,2), range_y=(-4, 4), 
             grid_size=50, color='Viridis', 
             loss=nn.CrossEntropyLoss(reduction='none'), z_by_loss=True, color_by_loss=False,
             x_ratio=1, y_ratio=1, z_ratio=0.8  
            )
```