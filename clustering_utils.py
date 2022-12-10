import os
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/" # for matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Serif TC"
plt.rcParams["axes.unicode_minus"] = False
from typing import Tuple, List
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

def plot_hierarchical_image(
    X: pd.DataFrame, 
    Y: List[str],
    save_path: str = None,
    plot_show: bool = False,
    method: str = "complete", 
    metric: str = "euclidean",
):
    fig = plt.figure(figsize=(8, 24), dpi=300)
    row_clusters = linkage(X.values, method=method, metric=metric)
    row_dendr = dendrogram(row_clusters, labels=Y, orientation="right")
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    # plt.xticks(rotation=-90)
    # plt.xticks(fontsize=8)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plot_show = True
    if plot_show:
        plt.show()