import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_2d_animation(coordinates:np.array, labels:np.array, attributes:np.array, markers:list, save_path:str=None):
    """Function to plot animation of points through time as a dynamic scatter plot

    Args:
        coordinates (np.array): array of shape n*2*t representing the coordinates (x,y) of n points at t time steps
        labels (np.array): 1D array of n labels one for each point
        attributes (np.array): 1D array of n attributes one for each point
        markers (list of strings): list of strings of markers to use, one for each attribute
        save_path (string): If given, the animation will be saved to this path. Default to None

    Returns:
        animation_plt: plt object of the animation
    """
    fig, ax = plt.subplots()

    # Get unique attributes
    unique_attributes = np.unique(attributes)
    
    t = coordinates.shape[2]

    # Create a dictionary to store scatter plot objects for each attribute
    scatter_plots = {}

    for attr in unique_attributes:
        # Filter tsne and labels based on attribute
        tsne_attr = coordinates[attributes == attr]
        labels_attr = labels[attributes == attr]

        # Scatter plot for current attribute
        scatter_plots[attr] = ax.scatter(tsne_attr[:, 0, 0], tsne_attr[:, 1, 0], s=50, marker=markers[attr], c=labels_attr, cmap='tab10')

    # ax.legend()

    def update(frame):
        for attr, scatter_plot in scatter_plots.items():
            # Filter tsne based on attribute and update scatter plot
            tsne_attr = coordinates[attributes == attr]
            x = tsne_attr[:, 0, frame]
            y = tsne_attr[:, 1, frame]
            data = np.stack([x, y]).T
            scatter_plot.set_offsets(data)

        return list(scatter_plots.values())

    animation_plt = animation.FuncAnimation(fig=fig, func=update, frames=t, interval=100)
    
    if save_path is not None:
        animation_plt.save(filename=save_path, writer="pillow")
    
    return animation_plt