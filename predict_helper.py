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


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    if(model == "vgg"):
        nhu = checkpoint['nhu']
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, nhu)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(nhu, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    elif(model == "densenet"):
        nhu = checkpoint['nhu']
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, nhu)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(nhu, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image = pil_image.resize((256, 256))

    width, height = pil_image.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    pil_image = pil_image.convert('RGB')
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image)


def predict(image_path, model, device="cpu", topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print(device)
    output_image = process_image(image_path).to(device)
    image = torch.zeros([64, 3, 224, 224], dtype=torch.float64).to(device)
    image += output_image.to(device)
    model.to(device)
    model.eval()
    torch.no_grad()
    logps = model.forward(image.float())
    ps = torch.exp(logps)
    probability, index = torch.topk(ps, topk, dim=1)
    return probability.to(device), index.to(device)


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
    parser.add_argument('input', type=str,
                        help='path to the image')
    parser.add_argument('ckpdir', type=str,
                        help='path to the folder of check point')

    parser.add_argument('--arch', type=str, default='vgg',
                        help='The Network architecture')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='gpu enable')
    parser.add_argument('--topk', type=float, default=5,
                        help='topk')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='directory of jason file')
    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()

    return in_args
