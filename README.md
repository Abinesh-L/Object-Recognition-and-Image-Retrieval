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

# Task 1
The goal is to design custom CNN architecture for object recognition on the Tiny ImageNet dataset. Since the test data of Tiny ImageNet doesn’t come with ground truth labels the validation data (which does have labels) is used as the testing data. After much experimentation, it is found that a shallower network performed better than a super deep one. The use of regularization (applied to each layer), dropouts, and data augmentation was necessary to avoid overfitting. After all of the experimentation, the custom model could reach a classification accuracy of 36.67% .

# Task 2
The goal is training from scratch a standard ResNet18 model on the Tiny ImageNet dataset. Using the open-source Keras implementation of ResNet (link), training on Tiny ImageNet led to a 44.27% classification accuracy.

# Task 3
In Task 3 the Resnet18 Model for Tiny ImageNet dataset is augmented using two CUBS block modules. These two CUBS blocks are integrated after the residual block in the Resnet18 model. Both CUBS1 and CUBS2 Block modules are designed using the given block diagram. For both blocks, cosine similarity is used to compute the similarity feature matrix. After integration of the CUBS blocks, the ResNet18 architecture is augmented in 3 different ways:
1)	Adding the summation of CUBS1 and CUBS2 modules after the residual block (CUBS-ResNet-1)
2)	Adding the CUBS1 module after the residual block (CUBS-ResNet-2)
3)	Adding the CUBS2 module after the residual block (CUBS-ResNet-3)
After training each augmented model on the Tiny ImageNet dataset, the obtained classification accuracies were 38.26%, 44.03%, and 43.62% respectively. Although not by much, it is shown here that CUBS-ResNet-2 performed the best, about on par with the base ResNet18 model.

# Task 4
In task 4, the goal is to use the best performing augmented CUBS ResNet architecture from task 3 to do image retrieval. The image retrieval task is to compute the similarity of a query image against a collection of other images, and retrieve the most similar one(s). It was determined that the second custom CUBS ResNet architecture (CUBS-ResNet-2) performed the highest in terms of classification accuracy in task 3, so that was the starting point for task 4. 

Using all cars196 dataset images split by class (98/98), the CUBS-ResNet-2 model is trained unmodified from task 3. A classification layer forms the end of this model, and as such the softmax probabilities are what is output. The recall @k function is a metric describing how many of the top similarities for a given sample lie above a threshold of interest. For all of task 4, the threshold is set at 0.75 and computed the similarity between two feature vectors using cosine similarity. With the “feature vectors” being that of the softmax probabilities, the expected recall at each K (1,2,4,8) is to shrink as the loss decreases. This is due to the model learning to predict classes of the 98 training images, and setting a large distinction between each prediction. These “feature vectors” in the eyes of the recall function should not be similar as the training loss decreases (and thus recall should decrease for all K). This was observed during training. The recall values started out at K1 = 0.003, K2 = 0.011, K4 = 0.040 and K8 = 0.197. As training resumed, each recall value quickly fell to 0.

Since the task is image retrieval, the final classification layer of the CUBS-ResNet-2 architecture is clipped and added a 512 dimensional feature embedding. This came in the form of a simple dense layer with 512 nodes. This embedding now lets the model output a 512 dimensional feature vector that represents the representation of a given image. This works well with the custom loss function, as the goal is to separate out and form groups of similar images on a hypersphere. By training with only the first 98 classes, the model now acts on data it’s never encountered before. By computing the embeddings, the loss function is able to “bring together” and “push away” other images depending upon how similar they are. With this in mind, the inverse effect was expected of what happened when training the base classification model for retrieval: as training loss decreases, each recall @K value should increase. This was observed during training, as the recall values started at K1 = 0.001, K2 = 0.003, K4 = 0.014 and K8 = 0.083, and quickly rose to K1 = 0.008, K2 = 0.023, K4 = 0.078 and K8 = 0.281. These values plateaued however, and stopped increasing even as the loss decreased. Although it is expected that the conceptual understanding of task 4 to be correct, it can be senn that there may be a gradient-related bug in the loss formulation that prevented the embedding layer from gaining any additional performance.
