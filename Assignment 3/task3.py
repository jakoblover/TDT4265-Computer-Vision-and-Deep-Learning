import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer

        # Define the convolutional layers
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=image_channels,
        #         out_channels=num_filters,
        #         kernel_size=2,
        #         stride=2,
        #         padding=0
        #     ),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=num_filters,
        #         out_channels=num_filters * 2,
        #         kernel_size=2,
        #         stride=2,
        #         padding=0
        #     ),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=num_filters * 2,
        #         out_channels=num_filters * 4,
        #         kernel_size=2,
        #         stride=2,
        #         padding=0
        #     ),
        #     nn.ReLU(),
        # )

        self.feature_extractor = nn.Sequential(
            #[32x32x3]
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

            #[16x16x32]
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_filters*2),
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_filters*2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4),

            #[8x8x64]
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_filters*4),
            nn.Conv2d(
                in_channels=num_filters * 4,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_filters*4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]

        # layer1_output = ((num_filters/2)-5+(2*2),(num_filters/2)-5+(2*2),32)
        # layer2_output = ((layer1_output[0]/2)- 5 + (2 * 2), (layer1_output[1]/2) - 5 + (2 * 2), 64)
        # layer3_output = ((layer2_output[0] / 2) - 5 + (2 * 2), (layer2_output[1] / 2) - 5 + (2 * 2), 128)

        # self.num_output_features = int(layer3_output[0]*layer3_output[1]*layer3_output[2])
        self.num_output_features = 4 * 4 * 128

        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )

        self.feature_extractor.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x

    def init_weights(self,a):
        if type(a) == (nn.Linear or nn.Conv2d):
            torch.nn.init.xavier_uniform_(a.weight)
            a.bias.data.fill_(0.01)


class Trainer:

    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.1
        self.early_stop_count = 4

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = ExampleModel(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4,8], gamma=0.5, last_epoch=-1)


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
            self.scheduler.step()


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