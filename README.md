# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Bash Example:

python train.py "flowers" --gpu True --arch "vgg" --lr .002 --epochs 6

python predict.py "flowers/test/1/image_06764.jpg" "vgg.pth" --arch "vgg" --gpu True 