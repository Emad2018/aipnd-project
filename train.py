from torchvision import transforms, datasets, models
import train_helper as thelp
import torch
import torch.nn as nn
import torch.optim as optim

in_args = thelp.get_input_args()
data_dir = in_args.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
                   "test": transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
                   "valid": transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])}

image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                  "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
                  "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
                  }

dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
               "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64),
               "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64)}
if in_args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (device == "cpu"):
        print("not found gpu,.... running on cpu")
else:
    device = "cpu"
if(in_args.arch == "densenet"):
    model = thelp.densenet121_FlowerModel()
else:
    model = thelp.vgg11_FlowerModel()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.lr)
print(in_args)
thelp.train_model(model, dataloaders, criterion,
                  optimizer, image_datasets, device=device, epochs=in_args.epochs)
