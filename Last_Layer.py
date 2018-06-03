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
import data_processing

###############################################################
# ConvNet as fixed feature extractor
# Here, we need to freeze all the network except the final layer.
# we need to set 'requires_grad == False' to freeze the parameters
# so that the gradients are not computed in 'backward()'..
#
# You can read more this in the documentation
# `here <http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
#



model_conv = torchvision.models.resnet101(pretrained=True)
for param in model_conv.parameters():
    param.requries_grad = False

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, 2)

criterion =nn.CrossEntropyLoss()

# observe that only parameters of final layer are being optimized as
# opoosed to before

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Dacay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# Tran and evaluate
model_conv = data_processing.train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,num_epochs=25)

#visualize_model(model_conv)

#plt.ioff()
#plt.show()
