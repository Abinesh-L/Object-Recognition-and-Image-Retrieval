# Recognition-and-Image-Retrieval
The objective of this project is to use a Deep Learning Framwork to create object recognition and retrieval systems. The primary goal of the
project is to do as well as possible on the image classification problem ( on the Tiny
ImageNet dataset) and on the image retrieval problem ( on the cars 196 dataset) using
convolutional neural networks.

#Datasets:
#TinyImagenet:
Tiny Imagenet has 200 classes. Each class has 500 training images, 50
validation images, and 50 test images. All images are 64 X 64. You can
download the dataset from the link below Link
http://cs231n.stanford.edu/tiny-imagenet-200.zip
#Cars196:
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split
into 8,144 training images and 8,041 testing images, where each class has been
split roughly in a 50-50 split. You can download the dataset from the link below
link
https://ai.stanford.edu/~jkrause/cars/car_dataset.html

The goal is to design custom CNN architecture for object recognition on the Tiny ImageNet dataset. Since the test data of Tiny ImageNet doesn’t come with ground truth labels the validation data (which does have labels) is used as the testing data. After much experimentation, it is found that a shallower network performed better than a super deep one. The use of regularization (applied to each layer), dropouts, and data augmentation was necessary to avoid overfitting. After all of the experimentation, the custom model could reach a classification accuracy of 36.67% .
