# TDT4265 Computer Vision and Deep Learning
This repo contains the assignments done in the computer vision and deep learning class at NTNU.

## Assignment 1
Assignment 1 focused on deriving the gradients for logistic regression and softmax regression, and implementing these from scratch using NumPy. We used the MNIST dataset to test the algorithms on handwritten digits.
First, a binary classification algorithm using logistic regression was created. Then, we introduce regularization to improve generalization. Finally, we implement a 10-way classification algorithm using softmax regression.

**Feedback**

1.2) You separated into two cases, but shouldn't there also be a sum over k'!=k at Eq. 15? The leap from 14 to 15 is quite big, so it's hard to see the process.

2.2d) If you zoom very closely, it is possible to see that the lower lambda is noisier, but I believe you probably already understand the effect of regularization :) (the effect would be more apparent if you didn't apply the learning rate to the L2 regularization, however, I would say both versions are appropriate.)


**Score: 5.625/6**

## Assignment 2
Assignment 2 focused on backpropagation, hidden layers, and known tricks to improve the performance of neural networks.
Tricks that are introduced are for example:
* Initializing weights using a normal distribution
* Improved sigmoid activation function
* Shuffle training set after each epoch
* Implementing momentum

**Feedback**

2b) Not implemented

4a) When reducing the number of nodes in the hidden layer, the concern of underfitting should be discussed.

4b) The concern of potential overfitting should be discussed. Only a small point-reduction since overfitting is mentioned in 4c.

**Score: 5.9/6**

## Assignment 3
Assignment 3 was the first task that involved convolutional neural networks. We used the CIFAR-10 dataset 
to perform 10-way classification on images. 
We used PyTorch as our deep learning framework, and compare how different parameters affect the performance of our network.
Finally, we perform transfer learning using ResNet18, trained on a much larger dataset, and compare it to the performance of our own networks.

**Feedback**

Great job! I think you have understood all concepts clearly, and the report is excellent. Keep it up! 

**Score: 6/6**

## Assignment 4
Assignment 4 is about YOLO (You Only Look Once) and metrics. 

**Feedback**

Great job! Unfortunately the result of your AP2 is not correct, from a quick glance at your task1_d code, I noticed you don't use idx_2, it’s probably a related coding error.

**Score: 5.85/6**
