"""
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def plot_graphs(graphs, titles, FILENAME="./", save=False, show=True):
    """ Plot n graphs with the specified titles
    :param graphs: list(np.array) graphs to be plot
    :param titles: list(string) titles of the graphs
    :return:
    """
    plt.subplots(figsize=(15,5))
    for i, (graph, title) in enumerate(zip(graphs, titles)):

        plt.subplot(1, len(graphs), i+1)
        plt.imshow(graph)
        plt.title(title)

    #plt.suptitle(titles[-1])
    plt.tight_layout()
    if save:
        plt.savefig(FILENAME)#, dpi=400)

    if show:
        plt.show()
    plt.close()



refinement = 'indeg_guided'
ksize = 23
imbalanced = False
num_c = 8
internoiselvl = 0
intranoiselvl = 0

G = np.random.rand(1000, 1000)

c = Codec(0, 0.5, 20)
k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
sze = c.decompress(G, 0, classes, k, reg_list)
fsze = c.post_decompression(sze, ksize)
red = c.reduced_matrix(G, k, epsilon, classes, reg_list)

plot_graphs([G, sze, fsze], ["G", f"sze:{k}", "fsze"])

ipdb.set_trace()
