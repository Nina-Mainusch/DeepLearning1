import torch
import matplotlib.pyplot as plt
from torch import nn


def weight_reset(model):
    """
    Resets all weights of a given model

    Args:
        model (nn.Module):        Model
    """
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        model.reset_parameters()


def build_kfold(data, k_folds):
    """
    Build indices for k-fold cross-validation.
    """
    N = len(data)
    perm_indices = torch.randperm(N)

    for k_iteration in range(k_folds):
        train_indices = perm_indices[torch.arange(N) % k_folds != k_iteration]
        val_indices = perm_indices[torch.arange(N) % k_folds == k_iteration]

        yield train_indices, val_indices


def test_performance(data, network, criterion, N, testing_loss=[], testing_accuracies=[], epoch=0, status=False,
                     learn_on_classes=False):
    """
    Calculates the testing accuracy and loss of a given model

    Args:
        data (torch.utils.data.Dataset):    Test data
        network (nn.Module):                Model
        criterion (nn.Object):              Loss function
        N (int):                            Number of test images
        testing_loss (list):                Testing loss
        testing_accuracies (list):          Testing accuracies
        epoch (int):                        Current epoch we are in
        status (bool):                      Indicator if current loss/accuracies should be printed to the console
        learn_on_classes (bool):            Indicator if model is learned on classes

    Returns:
        testing_loss (list):        Filled testing loss list
        testing_accuracies (list):  Filled testing accuracy list
    """
    if learn_on_classes:
        target_pred, output1, output2 = network(data.data)
        test_loss1 = criterion(output1, data.classes[:, 0])
        test_loss2 = criterion(output2, data.classes[:, 1])
        test_loss = test_loss1 + test_loss2

    else:
        target_pred, output = network(data.data)
        test_loss = criterion(output, data.target)

    nr_correct = torch.eq(target_pred, data.target).sum().item()

    if status:
        print(f"Test Loss of the model on the {N} test images: {test_loss:.4f}")
        print(f"Test Accuracy of the model on the {N} test images: {(nr_correct / N) * 100}%")

    else:
        testing_loss.append((epoch, test_loss.item()))
        testing_accuracies.append((epoch, nr_correct / N))

        return testing_loss, testing_accuracies


def plot_performance(training_loss, training_accuracies, training_stds, testing_loss, testing_accuracies):
    """
    Plots the MSE/accuracy over training and testing dataset over the epochs.

    Args:
        training_loss (list):           Training MSEs
        training_accuracies (list):     Training accuracies
        training_stds (list):           Training standard deviations of accuracies
        testing_loss (list):            Testing MSEs
        testing_accuracies (list):      Testing accuracies
    """
    plt.figure("Loss Summary")
    plt.plot([x[0] for x in training_loss], [x[1] for x in training_loss], label="Training Loss")
    plt.plot([x[0] for x in testing_loss], [x[1] for x in testing_loss], label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.figure("Accuracy Summary")
    plt.plot([x[0] for x in training_accuracies], [x[1] for x in training_accuracies], label="Training Accuracy")
    plt.plot([x[0] for x in testing_accuracies], [x[1] for x in testing_accuracies], label="Testing Accuracy")
    plt.fill_between([x[0] for x in training_accuracies],
                     [y[1] - std[1] for y, std in zip(training_accuracies, training_stds)],
                     [y[1] + std[1] for y, std in zip(training_accuracies, training_stds)], color='b', alpha=.1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()
