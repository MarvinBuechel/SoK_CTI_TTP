import torch
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# a few constants
SEED_VAL = 42
MAX_GRAD_NORM = 10
TAU = 0.5
PATIENCE = int(os.environ.get("PATIENCE", 15))



def parse_config(config):
    """
    Parses configuration file and returns id,model_name,dataset_name and model_params.
    id is just an identifier for the configuration.
    model_name must be in const.MODELS
    dataset_name must be in const.ExpDataset
    model_params can be 'batch_size','freeze_layers', and 'learning_rate'
    """
    with open(config, "r") as f:
        data = json.load(f)
    return [
        (x["id"], x["model_name"], x["dataset_name"], x["model_params"]) for x in data
    ]


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def make_report(output_dir, model_params, training_stats):
    """
    Generates and saves final results in folder 'output_dir'.
    """
    with open(os.path.join(output_dir, "model_params.json"), "w") as f:
        json.dump(model_params, f, indent=4)
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(training_stats, f, indent=4)
    plot_stats(output_dir, training_stats)


def plot_stats(output_dir, training_stats):
    """
    Plots with seaborn style the training stats of the model.
    The plot will show training and validation loss.
    """
    # Create a DataFrame from the training statistics
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index
    df_stats = df_stats.set_index("epoch")
    sns.set(style="darkgrid")

    # Increase the plot size and font size
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve
    plt.plot(df_stats["Training Loss"], "b-o", label="Training")
    plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")

    # Label the plot
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([i for i in range(1, len(df_stats["Training Loss"])+1)])
    plt.savefig(os.path.join(output_dir, "loss.pdf"), bbox_inches="tight")
    plt.clf()
