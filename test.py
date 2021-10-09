import torch
from data import P1Dataset
from networks import MLP, ConvNet, ConvTransposeNet, ConvResNet, ConvSiameseNetTargets, ConvSiameseResNet, \
    MNISTResNet, MNISTResSiameseNet, ConvSiameseNetClasses
from train import train_model

# Hyperparameters
MAX_NUM_EPOCHS = 50  # 25
PATIENCE = 10
FOLDS = 8  # 10
BATCH_SIZE = 128

# Load the data
N = 1000
train_data = P1Dataset(N, "train")
test_data = P1Dataset(N, "test")

# For reproducibility
torch.manual_seed(42)

if __name__ == "__main__":
    _ = train_model(MLP(), train_data=train_data, test_data=test_data, learn_on_classes=False,
                    show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                    folds=FOLDS, patience=PATIENCE)

    _ = train_model(ConvNet(), train_data=train_data, test_data=test_data, learn_on_classes=False,
                    show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                    folds=FOLDS, patience=PATIENCE)

    _ = train_model(ConvTransposeNet(), train_data=train_data,
                    test_data=test_data, learn_on_classes=False,
                    show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS,
                    batch_size=BATCH_SIZE, folds=FOLDS, patience=PATIENCE)

    _ = train_model(ConvResNet(nb_channels=15, kernel_size=3, nb_blocks=10),
                    train_data=train_data, test_data=test_data,
                    learn_on_classes=False, show_performance_plot=False,
                    max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                    folds=FOLDS, patience=PATIENCE)

    _ = train_model(ConvSiameseNetClasses(), train_data=train_data,
                    test_data=test_data, learn_on_classes=True,
                    show_performance_plot=False,
                    max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                    folds=FOLDS, patience=PATIENCE)

    _ = train_model(ConvSiameseNetTargets(), train_data=train_data,
                    test_data=test_data, learn_on_classes=False,
                    show_performance_plot=False,
                    max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                    folds=FOLDS, patience=PATIENCE)

    _ = train_model(MNISTResNet(), train_data=train_data, test_data=test_data,
                    show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS,
                    batch_size=BATCH_SIZE, folds=FOLDS, patience=PATIENCE)

    _ = train_model(MNISTResSiameseNet(), train_data=train_data,
                    test_data=test_data, show_performance_plot=False,
                    learn_on_classes=True, max_num_epochs=MAX_NUM_EPOCHS,
                    batch_size=BATCH_SIZE, folds=FOLDS, patience=PATIENCE)

    _ = train_model(
        ConvSiameseResNet(nb_channels=15, kernel_size=3, nb_blocks=10), train_data=train_data, test_data=test_data,
        learn_on_classes=True, show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
        folds=FOLDS, patience=PATIENCE)
