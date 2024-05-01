# viz_utils.py - Description:
#  Common tools for 3D/4D visualization.
# Created by Ruiming Cao on May 22, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ipywidgets import interact, IntSlider
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def volume(v, cmap='magma'):
    f, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    frames = []

    def init():
        frames.append(ax.imshow(v[0], clim=(np.min(v), np.max(v)), cmap=cmap, ))
        f.tight_layout()

    init()

    def updateFrames(z):
        frames[0].set_data(v[z])

    interact(updateFrames, z=IntSlider(min=0, max=v.shape[0] - 1, step=1, value=0))


def volume4d(v, cmap='magma'):
    f, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    frames = []

    def init():
        frames.append(ax.imshow(v[0, 0], clim=(np.min(v), np.max(v)), cmap=cmap, ))
        f.tight_layout()

    init()

    def updateFrames(t, z):
        frames[0].set_data(v[t, z])

    interact(updateFrames, z=IntSlider(min=0, max=v.shape[1] - 1, step=1, value=0),
             t=IntSlider(min=0, max=v.shape[0] - 1, step=1, value=0))


def add_scalebar(ax_, length, label, loc='upper left', pad=0.2, color='white', size_vertical=1, font_size=9,
                 label_top=True):
    scalebar = AnchoredSizeBar(ax_.transData,
                               length, label, loc,
                               pad=pad,
                               color=color,
                               frameon=False,
                               label_top=label_top,
                               size_vertical=size_vertical,
                               fontproperties=mpl.font_manager.FontProperties(size=font_size))
    ax_.add_artist(scalebar)


def add_inset(ax_, mat, extent):
    ax_in = ax_.inset_axes([0.54, 0.04, .47, .47], )
    frame = ax_in.imshow(mat, cmap='gray', clim=(0, 1), extent=extent, )
    x1, x2, y1, y2 = 27.8, 45.3, 0.6, 16.6  # 21.7, 39.2, 0.6, 16.6
    ax_in.set_xlim(x1, x2)
    ax_in.set_ylim(y1, y2)
    ax_in.set_xticks([],)
    ax_in.set_yticks([],)
    ax_in.tick_params(color='white')
    for spine in ax_in.spines.values():
        spine.set_edgecolor('white')

    ax_.indicate_inset_zoom(ax_in, alpha=1., edgecolor='white',lw=1)
    return frame, ax_in


def color_coded_projection(stack_3d):
    """
    Convert a 3D microscopy image stack to a color-coded 2D projection.

    Args:
    - stack_3d (numpy.ndarray): A 3D numpy array representing the image stack.

    Returns:
    - numpy.ndarray: A 2D color-coded projection image.
    """
    # Get the depth (number of slices) of the stack
    depth = stack_3d.shape[0]

    # Get a colormap
    colormap = mpl.cm.get_cmap('jet', depth)  # 'viridis' is just an example, you can choose any other colormap

    # Initialize an empty color image with the same width and height as a slice but with 3 channels for RGB
    color_stack = np.zeros((depth, stack_3d.shape[1], stack_3d.shape[2], 3))

    # Assign each slice a color and convert grayscale values to RGB using that color
    for z in range(depth):
        color = colormap(z)[:3]  # Get the RGB components of the color
        for i in range(3):  # For R, G, B channels
            color_stack[z, :, :, i] = stack_3d[z, :, :] * color[i]

    # Determine the indices of the maximum values in the original grayscale stack
    max_indices = np.argmax(stack_3d, axis=0)

    # Use the maximum indices to get the values from the color-coded stack
    projection = np.zeros((stack_3d.shape[1], stack_3d.shape[2], 3))
    for x in range(stack_3d.shape[1]):
        for y in range(stack_3d.shape[2]):
            projection[x, y] = color_stack[max_indices[x, y], x, y]

    return projection
