from __future__ import print_function, division
from configparser import Interpolation
from distutils.command.config import config
from pyexpat import model

# import torch and base libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import os
import copy
import json

# import mlflow 
import mlflow
from mlflow_utils import get_mlflow_experiment


from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms.functional import InterpolationMode

from base_model import MobNetv2_custom_classes


experiment = get_mlflow_experiment(experiment= "PyTorchClassifier")

cudnn.benchmark = True
# plt.ion()   # interactive mode

# loading configs
configs = json.load(open("config.json"))

# Initializing logger
writer = SummaryWriter()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224,transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224,transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = configs['train_dir_path']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=configs["batch_size"],
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                        # calculate gradients dy/dx
                        loss.backward()
                        # update weights
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # writing logs
            if phase == 'train':
                writer.add_scalar("Loss/train",epoch_loss,epoch)
                writer.add_scalar("Accuracy/train",epoch_acc,epoch)
            if phase == "val":
                writer.add_scalar("Loss/val",epoch_loss,epoch)
                writer.add_scalar("Accuracy/val",epoch_acc,epoch)


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



custom_mobilenet = MobNetv2_custom_classes()

# uncomment the below line to see the summary and sizes of each layer
# summary(custom_mobilenet,(3,224,224))


model_ft = custom_mobilenet.to(device)

loss_criteron = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=configs['learning_rate'], momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, loss_criteron, optimizer_ft, exp_lr_scheduler,
                       num_epochs=configs['epochs'])

torch.save(model_ft, configs["model_save_path"])
writer.close()