import torch

def to_cuda(elements):
    """
    Transfers elements to GPU memory, if a nvidia- GPU is available.
    Args:
        elements: A list or a single pytorch module.
    Returns:
        The same list transferred to GPU memory
    """

    if torch.cuda.is_available(): # Checks if a GPU is available for pytorch
        if isinstance(elements, (list, tuple)):
            return [x.cuda() for x in elements] # Transfer each index of the list to GPU memory
        return elements.cuda()
    return elements


def compute_loss_and_accuracy(dataloader, model, loss_criterion):
    """
    Computes the total loss and accuracy over the whole dataloader
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: nn.CrossEntropyLoss()
    Returns:
        [loss_avg, accuracy]: both scalar.
    """
    # Tracking variables
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0

    for (X_batch, Y_batch) in dataloader:
        # Transfer images/labels to GPU VRAM, if possible
        X_batch = to_cuda(X_batch)
        Y_batch = to_cuda(Y_batch)
        # Forward pass the images through our model
        output_probs = model(X_batch)
        # Compute loss
        loss = loss_criterion(output_probs, Y_batch)

        # Predicted class is the max index over the column dimension
        predictions = output_probs.argmax(dim=1).squeeze()
        Y_batch = Y_batch.squeeze()

        # Update tracking variables
        loss_avg += loss.item()
        total_steps += 1
        total_correct += (predictions == Y_batch).sum().item()
        total_images += predictions.shape[0]
    loss_avg = loss_avg / total_steps
    accuracy = total_correct / total_images
    return loss_avg, accuracy
