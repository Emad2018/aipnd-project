import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse


def vgg11_FlowerModel(nhu):
    model = models.vgg11(pretrained=True)
    model.name = "vgg"
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, nhu)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(nhu, 102)),
                              ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model


def densenet121_FlowerModel(nhu):
    model = models.densenet121(pretrained=True)
    model.name = "densenet"
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, nhu)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(nhu, 102)),
                              ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model


def save_model(model, path_name, nhu):

    checkpoint = {
        'nhu': nhu,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),

    }
    torch.save(checkpoint, path_name+'.pth')


def train_model(model, dataloaders, criterion, optimizer, image_datasets, nhu, epochs=5, print_every=5, device="cpu"):
    steps = 0
    prev_accuracy = 0
    running_loss = 0
    model.to(device)
    class_to_idx = image_datasets['train'].class_to_idx
    model.class_to_idx = {class_to_idx[k]: k for k in class_to_idx}
    for epoch in range(epochs):
        for inputs, labels in dataloaders["train"]:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
        if(accuracy > prev_accuracy):
            prev_accuracy = accuracy
            save_model(model, model.name, nhu)
            print("saving checkpoint")
    return model


def model_test(model, dataloaders, criterion, device="cpu"):
    test_loss = 0
    accuracy = 0
    model.eval()
    model.to(device)
    print(device)
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'flowers'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. GPU as --GPU with default value 'cpu'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: that's a path to a folder
    parser.add_argument('dir', type=str,
                        help='path to the folder of flowers')
    parser.add_argument('--arch', type=str, default='vgg',
                        help='The Network architecture')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='gpu enable')
    parser.add_argument('--lr', type=float, default=.003,
                        help='learning_rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='epochs')
    parser.add_argument('--nhu', type=int, default=512,
                        help='number of heddien unit')
    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()

    return in_args
