import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

LOG_FILE_NAME = "log.csv"

sns.set_context(context="paper", font_scale=0.68)
sns.set_style("white", {'font.family': 'serif'})

def moving_avg(x, y, window_size=1):
    if window_size == 1:
        return x, y
    moving_avg_y = np.convolve(y, np.ones(window_size) / window_size, 'valid') 
    return x[-len(moving_avg_y):], moving_avg_y

def plot_run(paths, name, ax=None, x_key="steps", y_keys=["eval/loss"], window_size=1, max_x_value=None, **kwargs):
    for path in paths:
        assert LOG_FILE_NAME in os.listdir(path), "Did not find log file, found " + " ".join(os.listdir(path))
    for y_key in y_keys:
        xs, ys = [], []
        for path in paths:
            df = pd.read_csv(os.path.join(path, LOG_FILE_NAME))
            if y_key not in df:
                print("[research] WARNING: y_key was not in run, skipping plot", path)
            x, y = moving_avg(df[x_key], df[y_key], window_size=window_size)
            assert len(x) == len(y)
            if max_x_value is not None:
                y = y[x <= max_x_value] # need to set y value first
                x = x[x <= max_x_value]
            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        plot_df = pd.DataFrame({x_key: xs, y_key: ys})
        label = name + " " + y_key if len(y_keys) > 1 else name
        ci = "sd" if len(paths) > 0 else None
        sns.lineplot(ax=ax, x=x_key, y=y_key, data=plot_df, sort=True, ci=ci, label=label, **kwargs)

def create_plot(paths, labels, ax=None, title=None, color_map=None, xlabel=None, ylabel=None, **kwargs):
    assert len(labels) == len(labels), "The length of paths must the same as the length of labels"
    ax = plt.gca() if ax is None else ax

    # Setup the color map
    if color_map is None:
        color_map = {labels[i]: i for i in range(len(labels))}
    for k in color_map.keys():
        if isinstance(color_map[k], int):
            color_map[k] = sns.color_palette()[color_map[k]]

    # Construct the plots
    for path, label in zip(paths, labels):
        if LOG_FILE_NAME not in os.listdir(path):
            run_paths = [os.path.join(path, run) for run in os.listdir(path)]
        else:
            run_paths = [path]
        plot_run(run_paths, label, ax=ax, color=color_map[label], **kwargs)

    ax.set_title(title, pad=1)
    ax.tick_params(axis='y', pad=-2, labelsize=5)
    ax.tick_params(axis='x', pad=-2, labelsize=5)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=0)

    sns.despine(ax=ax)


def plot_from_config(config_path):
    '''
    --- Configuration design for plot files ---
    title: null
    kwargs:
        xlabel: etc.
        ylabel: etc.
    color_map:
        method_1: idx
        method_2: idx
    
    grid_shape: (rows, cols)
    fig_size: (6, 3) etc. or null
    legend_pos: first
    use_subplot_titles: true
    
    plots:
        title_1:
            methods:
                method_1: path
                method_2: path
            kwargs:
            image: image path if we want to add an image

        title_2:
            methods:
                method_1: path
                method_2: path
            config:
            image: image path
    '''
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    grid_shape = config['grid_shape']

    # Note that grid shape is given as (rows, cols)
    assert len(config['plots']) == grid_shape[0] * grid_shape[1]
    figsize = (2*grid_shape[1], grid_shape[0]) if config.get('fig_size') is None else config.get('fig_size')

    legend_pos = config.get('legend_pos')
    assert legend_pos in {"first", "last", "bottom", None}
    if legend_pos == "first":
        legend_index = 0
    elif legend_pos == "last":
        legend_index = len(config['plots']) - 1
    else:
        legend_index = None
    
    fig, axes = plt.subplots(*grid_shape, figsize=figsize)

    # Determine if we should include xlabels or ylabels
    use_xlabels = any(['xlabel' in plot.get('kwargs', {}) for plot in config['plots'].values()])
    use_ylabels = any(['ylabel' in plot.get('kwargs', {}) for plot in config['plots'].values()])

    for i, (plot_title, plot_config) in enumerate(config['plots'].items()):
        y_index, x_index = i // grid_shape[1], i % grid_shape[1]
        ax = axes[y_index, x_index]

        paths, labels = list(plot_config['methods'].values()), list(plot_config['methods'].keys())
        plot_title = plot_title if config.get('use_subplot_titles') else None
        plot_kwargs = plot_config.get('kwargs', {}).copy()
        plot_kwargs.update(config['kwargs'])

        create_plot(paths, labels, ax, plot_title, color_map=config.get('color_map'), **plot_kwargs)

        if x_index != 0 and not use_ylabels:
            ax.set_ylabel(None)
        if y_index != grid_shape[0] - 1 and not use_xlabels:
            ax.set_xlabel(None)
        if i != legend_index:
            ax.get_legend().remove()

        # Check to see if we can place an image in the corner of the plot.
        if plot_config.get('image') is not None:
            import matplotlib.image as mpimg
            # use inset axes to create an inset image
            image_x = 0.75 * figsize[0] / grid_shape[1]
            axins = inset_axes(ax, width="33%", height="33%", loc=4, borderpad=0)
            image = mpimg.imread(plot_config['image'])
            axins.imshow(image)
            axins.axis('off')

    if config.get('title'):
        fig.suptitle(config.get('title'))

    # If the legend is set to the bottom do it here
    if legend_pos == "bottom":
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower left', ncol=len(handles), bbox_to_anchor=(0.25, -0.01))
        plt.tight_layout(pad=0, rect=(0, 0.05, 1, 1))
    else:
        plt.tight_layout(pad=0)
