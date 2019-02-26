import numpy as np
import matplotlib.pyplot as plt

validation_loss_best = np.load("task2_best_model_VALIDATION_LOSS.npy")
validation_loss_resnet = np.load("task3_resnet_VALIDATION_LOSS.npy")
train_loss_resnet = np.load("task3_resnet_TRAIN_LOSS.npy")
train_loss_best = np.load("task2_best_model_TRAIN_LOSS.npy")

plt.figure(figsize=(12, 8))
plt.title("Cross Entropy Loss")
plt.plot(train_loss_best, label="Network 2 - Train loss ")
plt.plot(validation_loss_best, label="Network 2 - Validation loss")
plt.plot(train_loss_resnet, label="Resnet18 - Train loss")
plt.plot(validation_loss_resnet, label="Resnet18 - Validation loss")
plt.legend()
plt.show()