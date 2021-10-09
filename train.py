import os

import torch
from torch import nn
from auxiliary import weight_reset, plot_performance, test_performance, build_kfold


def train_model(model, train_data, test_data, show_performance_plot=False, learn_on_classes=False, max_num_epochs=25,
                batch_size=64, learning_rate=0.001, folds=8, patience=3, verbose=1):
    """
    Calculates the testing accuracy and loss of a given model

    Args:
        model (nn.Module):                  Model
        train_data (np.array):              Training data
        test_data:                          Test data
        show_performance_plot (bool):       Indicator if performance should be plotted
        learn_on_classes (bool):            Indicator if model is trained on classes
        max_num_epochs (int):               Maximum number of epochs for the training
        batch_size (int):                   Training batch size
        learning_rate (float):              Learning rate for training the model
        folds (int):                        Folds for the k-fold cross validation
        patience (int):                     Early stopping patience
        verbose (

    Returns:
        model_results (pandas df):          Test and Training results in a dataframe for line plots
        validation_results (pandas df):     Validation data from cross validation for bar plot
    """

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists for collecting the losses and the accuracies
    training_loss = []
    training_accuracies = []
    training_stds = []
    testing_loss = []
    testing_accuracies = []

    # List for collecting the k-fold estimates
    k_fold_accuracies = []

    print(f"[*TRAINING {model.__name__}*]")

    for fold, (train_ids, val_ids) in enumerate(build_kfold(train_data, folds)):
        if verbose == 1:
            print(f"\tFOLD {fold + 1}/{folds}")
        # Reset the model for each fold
        model.apply(weight_reset)

        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   num_workers=8, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                 num_workers=8, sampler=val_subsampler)
        waited_epochs = 0
        lowest_val_loss = None
        for epoch in range(max_num_epochs):

            # Lists for collecting the loss and the accuracies, but only for this epoch
            training_loss_epoch = []
            nr_correct_epoch = 0
            len_data_epoch = 0
            accuracies_epoch = []

            for i, (images_batch, targets_batch, classes_batch) in enumerate(train_loader):

                # Run the forward pass
                if learn_on_classes:
                    targets_pred, output1, output2 = model(images_batch)
                    loss1 = criterion(output1, classes_batch[:, 0])
                    loss2 = criterion(output2, classes_batch[:, 1])
                    loss = loss1 + loss2
                else:
                    # Learn on targets
                    targets_pred, outputs = model(images_batch)
                    loss = criterion(outputs, targets_batch)

                training_loss_epoch.append(loss.item())

                # Backprop and Adam optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Save the training accuracies
                nr_correct_batch = (torch.eq(targets_pred, targets_batch)).sum().item()
                nr_correct_epoch += nr_correct_batch
                len_data_epoch += len(targets_batch)
                acc_batch = nr_correct_batch / len(targets_batch)
                accuracies_epoch.append(acc_batch)

            # Compute the mean accuracy and mse for this epoch and store it
            training_loss.append((epoch, torch.mean(torch.Tensor(training_loss_epoch)).item()))
            training_acc_epoch = nr_correct_epoch / len_data_epoch
            training_accuracies.append((epoch, training_acc_epoch))
            training_stds.append((epoch, torch.std(torch.Tensor(accuracies_epoch)).item()))

            if verbose == 1:
                print(f"\tEPOCH {epoch + 1}/{max_num_epochs} | LOSS: {loss.item():.4f} | "
                      f"ACCURACY: {(training_acc_epoch * 100):.2f}%")

            # Intermediate testing: calculate loss and accuracy on the test data with the current model
            with torch.no_grad():
                testing_loss, testing_accuracies = test_performance(test_data, model, criterion,
                                                                    len(test_data), testing_loss,
                                                                    testing_accuracies, epoch,
                                                                    learn_on_classes=learn_on_classes)

            _, fold_val_loss = evaluate_acc(model, criterion, val_loader, learn_on_classes)

            # Early stopping
            if lowest_val_loss is None or fold_val_loss < lowest_val_loss:
                lowest_val_loss = fold_val_loss
                waited_epochs = 0
                torch.save(model.state_dict(), 'data/best_model')
            else:
                waited_epochs += 1

            if waited_epochs == patience:
                model.load_state_dict(torch.load('data/best_model'))
                break

        # Get the k-fold accuracies
        fold_val_acc, _ = evaluate_acc(model, criterion, val_loader, learn_on_classes)

        # Add the mean validation accuracy for these batches
        k_fold_accuracies.append((fold, fold_val_acc))

    # Test the model
    model.eval()  # A model using dropout has to be set in “train” or “test” mode
    with torch.no_grad():
        test_performance(test_data, model, criterion, len(test_data), status=True, learn_on_classes=learn_on_classes)

    model.train()

    if show_performance_plot:
        # Plot training/test loss/accuracy
        plot_performance(training_loss, training_accuracies, training_stds, testing_loss, testing_accuracies)

    # Save the trained model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(model.state_dict(), './models' + f'/{model.__class__.__name__}.ckpt')

    if model.__name__:
        print(f"[+] {model.__name__} FINISHED")

    return {"model_name": model.__name__,
            "training_loss": training_loss,
            "training_accuracies": training_accuracies,
            "testing_loss": testing_loss,
            "testing_accuracies": testing_accuracies,
            "training_stds": training_stds,
            "k_fold_accuracies": k_fold_accuracies}


def evaluate_acc(model, criterion, data_loader, learn_on_classes):
    nr_correct = 0
    len_data = 0
    total_loss = 0
    with torch.no_grad():
        for i, (images_batch, targets_batch, classes_batch) in enumerate(data_loader):

            # Run the forward pass
            if learn_on_classes:
                targets_pred, output1, output2 = model(images_batch)
                loss1 = criterion(output1, classes_batch[:, 0])
                loss2 = criterion(output2, classes_batch[:, 1])
                loss_batch = loss1 + loss2
            else:
                targets_pred, outputs = model(images_batch)
                loss_batch = criterion(outputs, targets_batch)

            nr_correct_batch = (torch.eq(targets_pred, targets_batch)).sum().item()
            nr_correct += nr_correct_batch
            len_data += len(images_batch)
            total_loss += loss_batch.item()

        accuracy_data = nr_correct / len_data

        return accuracy_data, total_loss
