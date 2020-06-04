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
            
        elif data_name == "FashionMNIST" :
            self.train_data = dsets.FashionMNIST(root=root, 
                                                 train=True,
                                                 download=True, 
                                                 transform=transform_train)
            
            self.test_data = dsets.FashionMNIST(root=root, 
                                                train=False,
                                                download=True, 
                                                transform=transform_test)
            
        elif data_name == "SVHN" :
            self.train_data = MNISTM(root=root, 
                                     split='train',
                                     download=True,    
                                     transform=transform_train)
            
            self.test_data = MNISTM(root=root, 
                                    split='test',
                                    download=True, 
                                    transform=transform_test)
            
        elif data_name == "MNISTM" :
            self.train_data = MNISTM(root=root, 
                                     train=True,
                                     download=True,    
                                     transform=transform_train)
            
            self.test_data = MNISTM(root=root, 
                                    train=False,
                                    download=True, 
                                    transform=transform_test)
            
        elif data_name == "ImageNet" :
            self.train_data = ImageNet(root=root, 
                                       split='train',
                                       download=True,    
                                       transform=transform_train)
            
            self.test_data = ImageNet(root=root, 
                                      split='val',
                                      download=True, 
                                      transform=transform_test)
        
        elif data_name == "USPS" :
            self.train_data = USPS(root=root, 
                                   train=True,
                                   download=True,    
                                   transform=transform_train)
            
            self.test_data = USPS(root=root, 
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
            
    
"""
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
CREDIT: https://github.com/corenel
"""

import errno
import os

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
            
            self.targets = self.train_labels
            
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))
            
            self.targets = self.test_labels
            

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')