import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import torch



# Make sure that your file is like:
# file_folder:
#   train_folder:
#     train_class_i_folder
#     ...
#   test:
#     test_class_i_folder
#     ...


######################################################################
# Load Data
# ---------
# We will use torchvision and torch.utils.data packages for loading the
# data.
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.

data_dir = '~./hymenoptera_data'
# Your file path
# Here we use hymenoptera_data
# You can download at <https://download.pytorch.org/tutorial/hymenoptera_data.zip>


# Here we use torchvision.transforms.Compose to transform and normalize data
# ------
# CenterCrop(size): 将给定的PIL.Image进行中心切割，输出size，这里size可以是tuple(H,W)
# RandomCrop（size, padding = 0)
# RandomHorizontalFlip() 以p=0.5的概率随机水平反转

# Resize(size,interpolation=2)
# Scale(size): 将输入的PIL.Image重新改变大小成给定的'size'，其中size是最小边的边长
# 若H>W,则改变后图片大小是 (size*H/W, size)
# 224*224 的图片Scale(56)得到 （56，56）
# Scale 是size=int时的Resize退化版本。

# RandomResizedCrop（size, scale=(0.08,1),ratio=(H,W),interpolation=2)
# A crop of random size of the original size
# and a random aspect ratio of the original aspect ratio is made.
# This crop is finally resized to given size.


# Pad (padding, fill=0) 将给定图片的所有边用pad value填充。padding是填充像素范围，fill是值
# Normalize(mean,std): normalized_image = (image-mean)/std
# ToTensor: 把一个取值范围是[0,255]的PIL.Image 或者 shape为(H,W,C)的numpy.array
#           转换为 [C,H,W]的[0,1]的torch.FloadTensor
# ToPILImage

# Resnet input  224
# Intcpn input  299

data_transforms={
'train': transforms.Compose([
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]),
'val': transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]),
}



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                  for x in ['train','val']}

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle = True, num_workers=4)
                for x in ['train','val']}

dataset_size = {x: len(image_datasets[x]) for x in ['train','val']}


class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.


def imshow(inp, title=None):
    
    """imshow for Tensor"""
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp +mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


inputs, classes = next(iter(data_loaders['train']))

out = torchvision.utils.make_grid(inputs)

try:
    imshow(out, title=[class_names[x] for x in classes])
except Exception as e:
    print(e)
    
    

#######################################################################
# Training the model
# - Scheduling the learning rate
# - Saving the best model
# 'scheduler' is an LR scheduler object from 'torch.optim.lr_scheduler'

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """

    :param model:
    :param criterion: calculate loss
    :param optimizer:
    :param scheduler: LR scheduler
    :param num_epochs:
    :return:
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        print('-'*10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase =='train':
                scheduler.step()  # learning rate 迭代一次
                model.train() # set model to training mode
            else:
                model.eval()    # set model to val mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        # 每训练完一个batch，更新一次系数权重算作一次iteration
        # training data 10w，batch=4，iteration=10w/4
        # 即一个epoch中要做2.5w次权重更新
        for inputs, labels in data_loaders[phase]:
           # inputs = inputs.to('cpu')
           # labels = labels.to('cpu')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase =='train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs,labels)

                # backward + optimizer only if in training phase
                if phase =='train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size[phase]
        epoch_acc = running_corrects.double() / dataset_size[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

####################################################
# - visualize your model

def visualize_model(model, num_images = 6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders['val']):
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(input.size()[0]):
                images_so_far +=1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode = was_training)
                    return
        model.train(mode = was_training)
        
        






