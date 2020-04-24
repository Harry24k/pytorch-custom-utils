import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

class Datasets() :
    def __init__(self, data_name, root='./data', val_idx=None,
                 label_filter=None,
                 transform_train=transforms.ToTensor(), 
                 transform_test=transforms.ToTensor(), 
                 transform_val=transforms.ToTensor()) :

        self.val_idx = val_idx
        
        # TODO : Validation + Label filtering
        if val_idx is not None :
            if label_filter is not None :
                raise ValueError("Validation + Label filtering is not supported yet.")
        
        # Load dataset
        if data_name == "CIFAR10" :
            self.train_data = dsets.CIFAR10(root=root, 
                                            train=True,
                                            download=True,    
                                            transform=transform_train)

            self.test_data = dsets.CIFAR10(root=root, 
                                           train=False,
                                           download=True, 
                                           transform=transform_test)
            
        elif data_name == "CIFAR100" :
            self.train_data = dsets.CIFAR100(root=root, 
                                             train=True,
                                             download=True, 
                                             transform=transform_train)

            self.test_data = dsets.CIFAR100(root=root, 
                                            train=False,
                                            download=True, 
                                            transform=transform_test)
            
        elif data_name == "STL10" :
            self.train_data = dsets.STL10(root=root, 
                                          split='train',
                                          download=True, 
                                          transform=transform_train)
            
            self.test_data = dsets.STL10(root=root, 
                                         split='test',
                                         download=True, 
                                         transform=transform_test)
            
        elif data_name == "MNIST" :
            self.train_data = dsets.MNIST(root=root, 
                                          train=True,
                                          download=True,    
                                          transform=transform_train)
            
            self.test_data = dsets.MNIST(root=root, 
                                         train=False,
                                         download=True, 
                                         transform=transform_test)
            
        else : 
            raise ValueError(data_name + " is not valid")
            
        self.data_name = data_name
            
        if self.val_idx is not None :
            self.val_data = Subset(self.train_data, self.val_idx)            
            self.val_data.transform = transform_val            
            
            train_idx = list(set(range(len(self.train_data))) - set(self.val_idx))
            
            self.train_data = Subset(self.train_data, train_idx)
            
            self.val_len = len(self.val_idx)
            self.train_len = len(self.train_data)
            self.test_len = len(self.test_data)
            
            print("Data Loaded (w/ Validation Set)!")
            print("Train Data Length :", self.train_len)
            print("Val Data Length :", self.val_len)
            print("Test Data Length :", self.test_len)
            
        elif label_filter is not None :
            filtered = []
            
            # Tensor label to list
            if type(self.train_data.targets) is torch.Tensor :
                self.train_data.targets = self.train_data.targets.numpy()
            if type(self.test_data.targets) is torch.Tensor :
                self.test_data.targets = self.test_data.targets.numpy()
            
            for (i, label) in enumerate(self.train_data.targets) :
                if label in label_filter.keys() :
                    filtered.append(i)
                    self.train_data.targets[i] = label_filter[label]
            
            self.train_data = Subset(self.train_data, filtered) 
            self.train_len = len(self.train_data)
            
            filtered = []
            for (i, label) in enumerate(self.test_data.targets) :
                if label in label_filter.keys() :
                    filtered.append(i)
                    self.test_data.targets[i] = label_filter[label]    
                    
            self.test_data = Subset(self.test_data, filtered) 
            self.test_len = len(self.test_data)
            
            print("Data Loaded! (w/ Label Filtering)")
            print("Train Data Length :", self.train_len)
            print("Test Data Length :", self.test_len)
            
        else :
            self.train_len = len(self.train_data)
            self.test_len = len(self.test_data)
        
            print("Data Loaded!")
            print("Train Data Length :", self.train_len)
            print("Test Data Length :", self.test_len)
        
    def get_len(self) :
        if self.val_idx is None :
            return self.train_len, self.test_len

        else :
            return self.train_len, self.val_len, self.test_len
    
    def get_data(self) :
        if self.val_idx is None :
            return self.train_data, self.test_data

        else :
            return self.train_data, self.val_data, self.test_data
    
    def get_loader(self, batch_size) :
        
        if self.val_idx is None :
            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)

            self.test_loader = DataLoader(dataset=self.test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

#             print("Train Loader Length :", len(self.train_loader))
#             print("Test Loader Length :", len(self.test_loader))

            return self.train_loader, self.test_loader

        else :    
            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)

            self.val_loader = DataLoader(dataset=self.val_data,
                                         batch_size=batch_size,
                                         shuffle=False)

            self.test_loader = DataLoader(dataset=self.test_data,
                                          batch_size=batch_size,
                                          shuffle=False)
            
#             print("Train Loader Length :", len(self.train_loader))
#             print("Val Loader Length :", len(self.val_loader))
#             print("Test Loader Length :", len(self.test_loader))

            return self.train_loader, self.val_loader, self.test_loader          
            