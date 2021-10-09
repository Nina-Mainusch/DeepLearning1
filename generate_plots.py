import torch
import pandas as pd
from data import P1Dataset
from networks import MLP, ConvNet, ConvTransposeNet, ConvResNet, ConvSiameseNetTargets, ConvSiameseResNet, \
    MNISTResNet, MNISTResSiameseNet, ConvSiameseNetClasses
from train import train_model
from auxiliary_plots import plot_all_performances, plot_model_accuracies

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


def generate_dfs(training_output):
    # Save the data for a general plot
    df_1 = pd.DataFrame(training_output["training_loss"], columns=['epoch', 'train_loss'])
    df_2 = pd.DataFrame(training_output["training_accuracies"], columns=['epoch', 'train_acc'])
    df_3 = pd.DataFrame(training_output["testing_loss"], columns=['epoch', 'test_loss'])
    df_4 = pd.DataFrame(training_output["testing_accuracies"], columns=['epoch', 'test_acc'])
    df_5 = pd.DataFrame(training_output["training_stds"], columns=['epoch', 'train_std'])

    # Merge all data frames into a single one
    model_results = df_1.merge(df_2, left_on="epoch", right_on="epoch") \
        .merge(df_3, left_on="epoch", right_on="epoch") \
        .merge(df_4, left_on="epoch", right_on="epoch") \
        .merge(df_5, left_on="epoch", right_on="epoch")
    model_results["model"] = training_output["model_name"]
    model_results = model_results.groupby(["model", "epoch"]).mean().reset_index()

    # Validation results for the bar plot
    validation_results = pd.DataFrame(training_output["k_fold_accuracies"], columns=['fold', 'val_accuracies'])
    validation_results["model"] = training_output["model_name"]
    validation_results = validation_results.groupby(["model", "fold"]).mean().reset_index()

    return model_results, validation_results


if __name__ == "__main__":
    MLP_training_output = train_model(MLP(), train_data=train_data, test_data=test_data, learn_on_classes=False,
                                      show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                                      folds=FOLDS, patience=PATIENCE, verbose=0)
    MLP_pd, MLP_val = generate_dfs(MLP_training_output)

    ConvNet_training_output = train_model(ConvNet(), train_data=train_data, test_data=test_data, learn_on_classes=False,
                                          show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                                          folds=FOLDS, patience=PATIENCE, verbose=0)
    ConvNet_pd, ConvNet_val = generate_dfs(ConvNet_training_output)

    ConvTransposeNet_training_output = train_model(ConvTransposeNet(), train_data=train_data,
                                                   test_data=test_data, learn_on_classes=False,
                                                   show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS,
                                                   batch_size=BATCH_SIZE, folds=FOLDS, patience=PATIENCE, verbose=0)
    ConvTransposeNet_pd, ConvTransposeNet_val = generate_dfs(ConvTransposeNet_training_output)

    ConvResNet_training_output = train_model(ConvResNet(nb_channels=15, kernel_size=3, nb_blocks=10),
                                             train_data=train_data, test_data=test_data,
                                             learn_on_classes=False, show_performance_plot=False,
                                             max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                                             folds=FOLDS, patience=PATIENCE, verbose=0)
    ConvResNet_pd, ConvResNet_val = generate_dfs(ConvResNet_training_output)

    ConvSiameseNetClasses_training_output = train_model(ConvSiameseNetClasses(), train_data=train_data,
                                                        test_data=test_data, learn_on_classes=True,
                                                        show_performance_plot=False,
                                                        max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                                                        folds=FOLDS, patience=PATIENCE, verbose=0)
    ConvSiameseNetClasses_pd, ConvSiameseNetClasses_val = generate_dfs(ConvSiameseNetClasses_training_output)

    ConvSiameseNetTargets_training_output = train_model(ConvSiameseNetTargets(), train_data=train_data,
                                                        test_data=test_data, learn_on_classes=False,
                                                        show_performance_plot=False,
                                                        max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE,
                                                        folds=FOLDS, patience=PATIENCE, verbose=0)
    ConvSiameseNetTargets_pd, ConvSiameseNetTargets_val = generate_dfs(ConvSiameseNetTargets_training_output)

    MNISTResNet_training_output = train_model(MNISTResNet(), train_data=train_data, test_data=test_data,
                                              show_performance_plot=False, max_num_epochs=MAX_NUM_EPOCHS,
                                              batch_size=BATCH_SIZE, folds=FOLDS, patience=PATIENCE, verbose=0)
    MNISTResNet_pd, MNISTResNet_val = generate_dfs(MNISTResNet_training_output)

    MNISTResSiameseNet_training_output = train_model(MNISTResSiameseNet(), train_data=train_data,
                                                     test_data=test_data, show_performance_plot=False,
                                                     learn_on_classes=True, max_num_epochs=MAX_NUM_EPOCHS,
                                                     batch_size=BATCH_SIZE, folds=FOLDS, patience=PATIENCE, verbose=0)
    MNISTResSiameseNet_pd, MNISTResSiameseNet_val = generate_dfs(MNISTResSiameseNet_training_output)

    ConvSiameseResNet_training_output = train_model(ConvSiameseResNet(nb_channels=15, kernel_size=3, nb_blocks=10),
                                                    train_data=train_data, test_data=test_data,
                                                    learn_on_classes=True, show_performance_plot=False,
                                                    max_num_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE, folds=FOLDS,
                                                    patience=PATIENCE, verbose=0)
    ConvSiameseResNet_pd, ConvSiameseResNet_val = generate_dfs(ConvSiameseResNet_training_output)

    # Create a dataframe with all model's training and test accuracies and stds
    df = MLP_pd.append(ConvSiameseNetClasses_pd).append(ConvSiameseNetTargets_pd).append(MNISTResSiameseNet_pd)

    df_val = MLP_val.append(ConvNet_val).append(ConvSiameseNetClasses_val).append(ConvSiameseNetTargets_val) \
        .append(MNISTResNet_val).append(MNISTResSiameseNet_val).append(ConvResNet_val) \
        .append(ConvSiameseResNet_val)  # .append(ConvTransposeNet_val)

    # Create a plot of all models and accuracies
    plot_all_performances(df, save=True, filename='./plots/lineplot_model_accuracies')
    plot_model_accuracies(df_val, save=True, filename='./plots/barplot_val_accuracies')
