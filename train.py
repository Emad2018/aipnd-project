import torch
import os
from torchvision import datasets, transforms,models
import argparse
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, help='Path to the save directory', default='../../checkpoints/checkpoint.pth')
    parser.add_argument('--arch', type=str, help='Model architecture', default='vgg13')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=5)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

def save_checkpoint(model, optimizer, save_dir='checkpoint.pth', idx_to_class=None):
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': idx_to_class
    }, save_dir)

def validation(model, valloader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            val_loss += criterion(output, labels).item()
            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
    return val_loss, accuracy

def train(model, trainloader, valloader, criterion, optimizer, epochs=5, device='cpu', print_every=40):
    model.to(device)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    val_loss, accuracy = validation(model, valloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(val_loss/len(valloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

def getdataloader(data_dir, batch_size=32, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'valid', 'test']}
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
                for x in ['train', 'valid', 'test']}
    return dataloaders

def createModel(architecture, num_classes):
    if architecture not in models.__dict__:
        raise ValueError(f"Invalid model architecture: {architecture}")
    model = getattr(models, architecture)(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(512, 256)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(256, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return model

if __name__ == "__main__":
    args = parse_args()
    # Check if the architecture is valid
    if args.arch not in models.__dict__:
        raise ValueError(f"Invalid model architecture: {args.arch}")
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = getdataloader(args.data_dir, batch_size=32, num_workers=4)
    model = createModel(args.arch, num_classes=len(dataloaders['train'].dataset.classes))
    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, epochs=args.epochs, device=device, print_every=40)
    # Get class names from train_data
    idx_to_class = {v: k for k, v in dataloaders['train'].dataset.class_to_idx.items()}
    # Save the model checkpoint
    save_checkpoint(model, optimizer, save_dir=args.save_dir, idx_to_class=idx_to_class)
#python train.py  '../../Datasets/102flowers' --save_dir '../../checkpoints/flowersvgg13.pth' --epochs 1  --gpu