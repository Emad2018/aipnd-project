# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.
# Download the flower dataset using this link:
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
# Bash Example:

python train.py "flowers" --gpu True --arch "vgg" --lr .002 --epochs 6

python predict.py "flowers/test/1/image_06764.jpg" "vgg.pth" --arch "vgg" --gpu True 
