import torch
from torchvision import  transforms,models
import argparse
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('image_path', type=str, help='Path to the image file', default='image.jpg')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, help=' top K most likely classes', default=5)
    parser.add_argument('--arch', type=str, help='Model architecture', default='vgg13')
    parser.add_argument('--category_names', type=str, help='Path to the category names file', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(architecture,filepath,device):
    checkpoint = torch.load(filepath, map_location=device)
    model = getattr(models, architecture)(weights="DEFAULT")
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(512, 256)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['class_to_idx']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Resize the image to 256x256
    image = image.resize((256, 256))
    # Center crop the image to 224x224
    image = image.crop((16, 16, 240, 240))
    # Normalize the image
    image = transforms.functional.to_tensor(image)
    image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return np.array(image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.exp(output)
        top_probs, top_classes = probs.topk(topk, dim=1)
    return top_probs, top_classes

def display_prediction(image_path, model, idx_to_class, category_names, topk=5):
    image = Image.open(image_path)
    top_probs, top_classes = predict(image_path, model, topk)
    
    # Convert class indices to class names
    top_classes = top_classes.cpu().numpy()[0]
    # Invert the class_to_idx dictionary to get idx to class mapping
    top_classes = [idx_to_class[idx] for idx in top_classes]
    print(f"Classes: {top_classes}")
    class_names = [category_names[str(i)] for i in top_classes]

    # Display the image
    plt.figure(figsize=(6, 9))
    ax = plt.subplot(2, 1, 1)
    imshow(process_image(image), ax=ax)
    ax.set_title(class_names[0])
    
    # Display the top 5 classes and their probabilities 
    ax = plt.subplot(2, 1, 2)
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, top_probs[0].cpu().numpy(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    model, _, idx_to_class = load_checkpoint(args.arch, args.checkpoint, device)

    display_prediction(args.image_path, model, idx_to_class, cat_to_name, args.top_k)

#python predict.py "/home/emad/main/Repos/ML/Datasets/102flowers/test/18/image_04275.jpg"  "/home/emad/main/Repos/ML/checkpoints/flowersvgg13.pth" --gpu --top_k 6 --arch vgg13 --category_names "cat_to_name.json"