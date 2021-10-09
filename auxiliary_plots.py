import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prettytable import PrettyTable


def plot_all_performances(df, filename="./plots/test", save=False):
    """
    Plots the accuracy over training and testing data over the epochs for all models.
    Args:
        df (pandas df):           dataframe of all models and their accuracies
        filename (str):           path for saving the plot
        save (bool):              flag whether to save the figure
    """
    # Refactor the df for plotting
    df = df.rename(columns={'train_acc': 'Training', 'test_acc': 'Test', 'model': 'Model'})
    df_melt = pd.melt(df, id_vars=['epoch', 'Model'], value_vars=['Training', 'Test'], value_name="accuracy",
                      var_name="Accuracy Type")

    plt.figure(figsize=(10, 8))

    # Change the default layout
    sns.set(font_scale=1.3)
    sns.set_style("white")

    # Create the line plot
    lines = sns.lineplot(data=df_melt, x="epoch", y="accuracy", hue="Model", style="Accuracy Type")
    lines.set(xlabel='Epoch', ylabel='Accuracy')
    plt.legend(loc='lower right')

    if save:
        plt.tight_layout()
        plt.savefig(f"{filename}.pdf", format="pdf")

    plt.show()


def plot_model_accuracies(df, filename="./plots/test", save=False):
    """
    Plots the estimated mean and stds of the cross validation in a bar plot

    Args:
        df (pandas df):           dataframe of all models and their accuracies
        filename (str):           path for saving the plot
        save (bool):              flag whether to save the figure

    """
    # Sort the values in ascending order
    df = df.iloc[(df.groupby('model')['val_accuracies'].transform('mean')).argsort()]

    plt.figure(figsize=(10, 8))

    # Change the default layout
    sns.set(font_scale=1.3)
    sns.set_style("white")

    # Create the bar plot
    bars = sns.barplot(x="model", y="val_accuracies", data=df)
    bars.set_xticklabels(bars.get_xticklabels(),
                         rotation=15,
                         horizontalalignment='right')
    bars.set(xlabel='Model', ylabel='Validation Accuracy')
    bars.set(ylim=(.5, 1.0))

    if save:
        plt.tight_layout()
        plt.savefig(f"{filename}.pdf", format="pdf")

    plt.show()


# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model#:~:text=To%20get%20the%20parameter%20count,name%20and%20the%20parameter%20itself.
def count_parameters(model):
    """
    Counts number of parameters of a given model

    Args:
        model (nn.Module):        Model
    Returns:
        total_params (int):       Total number of parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
