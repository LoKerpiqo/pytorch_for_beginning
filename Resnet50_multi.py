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
import argparse
import shutil
import warnings
warnings.filterwarnings('ignore')


from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description = 'ResNet')
parser.add_argument('--resume',default = '',type=str,metavar='PATH',help='path to latest chackpoint(default:none)')
parser.add_argument('-b','--batch_size',default=64,type=int,metavar='N',help='mini-batch size(default:64)')
parser.add_argument('-a','--arch',default='resnet50',choices=model_names,
                    help='model architecture:'+'|'.join(model_names)+'(default:resnet50)')
parser.add_argument('--num_epochs','-e',default=25,type=int,metavar='N',help='training epoch (default:25)')
parser.add_argument('--data_dir',metavar='DIR',help='path to dataset')
parser.add_argument('--lr',default=0.01,type=float,metavar='LR',help='initial learning rate (default=0.01)')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manually epoch number(useful on resrats)')

best_acc=0


def main():
    global args, best_acc
    args = parser.parse_args()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = args.data_dir

    image_datasets = {x: myImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_fn)
    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    print("=>using pretrained model'{}'".format(args.arch))
    model_conv = models.__dict__[args.arch](pretrained=True)


    for param in model_conv.parameters():
        param.requries_grad = False

    num_ftrs = model_conv.fc.in_features

    model_conv.fc = nn.Linear(num_ftrs, 200)


    criterion = nn.CrossEntropyLoss().cuda()

    # observe that only parameters of final layer are being optimized as
    # opoosed to before
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Dacay LR by a factor of 0.1 every 4 epochs

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=4, gamma=0.1)

    from torch.nn import DataParallel
    model_conv = DataParallel(model_conv, device_ids=[0, 1]).cuda()


    train_model(dataloaders,dataset_sizes,model_conv, criterion, optimizer_conv, exp_lr_scheduler, args.num_epochs)



def train_model(dataloaders,dataset_sizes,model, criterion, optimizer, scheduler, num_epochs):
    global args, best_acc
    since = time.time()
    #num_epochs=args.num_epochs

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {} )".format(args.resume,checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch,args.num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # save_checkpoint the model
            # We focus on train dataset. Do not use 'validation' to select model.

            if phase == 'train' :
                is_best = epoch_acc > best_acc
                best_acc = max(epoch_acc,best_acc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights

    #model.load_state_dict(best_model_wts)
    #return model
    
    
def save_checkpoint(state, is_best, filename='hy_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'hy_model_best.pth.tar')


class myImageFolder(ImageFolder):
    __init__ = ImageFolder.__init__

    def __getitem__(self, index):
        try:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            return super(myImageFolder, self).__getitem__(index)
        except Exception as e:
            print(e)

def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch)

if __name__ =='__main__':
    main()

