import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
import numpy as np


class Model (nn.Module):
    def __init__(self):
        super ().__init__()
        self.model = torchvision.models.resnet18(pretrained = True)
        self.model.fc = nn.Linear(512*4, 10) # No need to apply softmax, as this is done in nn. C r o s s E n t r o p y L o s s
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully - connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 c on vo lu ti on al
            param.requires_grad = True # layers

    def forward (self, x):
        x = nn.functional.interpolate(x, scale_factor=8)
        x = self.model(x)
        return x


class Trainer:

    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.early_stop_count = 4

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = Model()
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            print("Epoch {0}".format(epoch))
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                # Compute loss/accuracy for all three datasets.
                if batch_it % self.validation_check == 0:
                    self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        return



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()

    print("Final train loss:", trainer.TRAIN_LOSS[-1])
    print("Final validation loss:", trainer.VALIDATION_LOSS[-1])
    print("Final train accuracy:", trainer.TRAIN_ACC[-1])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-1])
    print("Final test accuracy:", trainer.TEST_ACC[-1])

    # # Save loss and accuracy
    # np.save("TRAIN_LOSS.npy", np.array(trainer.TRAIN_LOSS))
    # np.save("VALIDATION_LOSS.npy", np.array(trainer.VALIDATION_LOSS))
    # np.save("TEST_LOSS.npy", np.array(trainer.TEST_LOSS))
    # np.save("VALIDATION_ACC.npy", np.array(trainer.VALIDATION_ACC))
    # np.save("TEST_ACC.npy", np.array(trainer.TEST_ACC))


    # # Visualize output from first convolutional layer
    # image = plt.imread("data/ship7.png")
    # image = torchvision.transforms.functional.to_tensor(image)
    # image = torchvision.transforms.functional.normalize(image.data, mean, std)
    # image = image.view(1, *image.shape)
    # image = nn.functional.interpolate(image, size=(256, 256))
    #
    # model = torchvision.models.resnet18(pretrained=True)
    # first_layer_out = model.conv1(image)
    # to_visualize = first_layer_out.view(first_layer_out.shape[1], 1, *first_layer_out.shape[2:])
    # torchvision.utils.save_image(to_visualize, "filters_first_layer.png")
    #
    # to_visualize = first_layer_out.view(first_layer_out.shape[1], 1, *first_layer_out.shape[2:])
    # torchvision.utils.save_image(to_visualize, "filters_first_layer.png")



    # #Forward pass through all layers except the fully connected layer
    # model = list(trainer.model.children())[0]
    # image = plt.imread("data/ship7.png")
    # image = to_cuda(torchvision.transforms.functional.to_tensor(image))
    # image = torchvision.transforms.functional.normalize(image.data, mean, std)
    # image = image.view(1, *image.shape)
    # image = nn.functional.interpolate(image, size=(256, 256))
    #
    # filter_img = model.conv1(image)
    # filter_img = model.bn1(filter_img)
    # filter_img = model.relu(filter_img)
    # filter_img = model.maxpool(filter_img)
    # filter_img = model.layer1(filter_img)
    # filter_img = model.layer2(filter_img)
    # filter_img = model.layer3(filter_img)
    # filter_img = model.layer4(filter_img)
    #
    # to_visualize = filter_img.view(filter_img.shape[1], 1, *filter_img.shape[2:])[:128]
    # torchvision.utils.save_image(to_visualize, "filters_last_layer.png")



