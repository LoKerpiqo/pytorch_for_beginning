
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

########################################################
#Finetuning the convenet

#Load a pre-trained model and reset final fully connected layer.



model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train and evaluate
# It should take around 15-25 min on cpu. It takes less than a minute.
# 250 training 150 val
model_ft = data_processing.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
#data_processing.visualize_model(model_ft)






