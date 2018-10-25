import os

import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Load the data into a DataFrame. The data file is located in ./data/
    """
    directory = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(directory, 'data', filename)
    data = pd.read_csv(filepath)

    return data


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'image'
    )
    if not os.path.isdir(directory):
        os.mkdir(directory)
    fig_path = os.path.join(directory, fig_id + '.' + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
