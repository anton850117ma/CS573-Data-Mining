import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


fig_num = 0


def plot(fname, x, *ys, figsize=None, x_label=None, y_label=None, legend=None, legend_loc=None, title=None):
    """
    Plot x VS y and save the plot to a file with fname.
    :param x: x
    :param y: y
    :param fname: file name which we want to save the plot to
    :param fmt: format of the plot
    :param figsize: figure size. Must be tuple (horizontal, vertical)
    :param x_label: label of x axis
    :param y_label: label of y axis
    """
    global fig_num
    plt.figure(fig_num, figsize=figsize)
    i = 0
    for y in ys:
        plt.plot(x, y, "C" + str(i) + "-o", ms=3)
        i += 1

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if legend is not None:
        if legend_loc is None:
            plt.legend(legend)
        else:
            plt.legend(legend, loc=legend_loc)
    plt.title(title)
    plt.savefig(fname, format="eps")
    fig_num += 1


def bar(fname, *ys, x_ticks=None, total_width=None, x_label=None, y_label=None, legend=None, legend_loc=None, title=None):
    """
    Draw bar graph and save graph to a file with fname.
    :param x_ticks: The tick labels of the bars. (labels on x axis at each index)
    :param fname: file name which to save bar graph to
    :param ys: y values for different categories
    :param total_width: total width of all bars at certain index (maximum 1)
    :param x_label: label of x axis
    :param y_label: label of y axis
    :param legend: legend for different categories
    :param legend_loc: legend location. Possible choices are "upper right" "upper left" "lower left" "lower right"
    """
    global fig_num
    plt.figure(fig_num)
    assert len(ys) >= 1
    n_categories = len(ys)
    x_indices = None
    if total_width is None:
        width = 0.8 / n_categories
    else :
        width = total_width / n_categories
    i = 0
    for y in ys:
        if x_indices is None:
            x_indices = np.arange(len(y))
        assert x_indices.shape[0] == len(y)
        plt.bar(x_indices + width * (i + 0.5 * (1 - n_categories)), np.array(y), width)
        i += 1
    plt.xticks(x_indices, x_ticks)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if legend is not None:
        if legend_loc is None:
            legend_loc = "upper right"
        plt.legend(legend, loc=legend_loc)
    plt.savefig(fname, format="eps")
    fig_num += 1
