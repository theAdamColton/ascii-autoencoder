import matplotlib.pyplot as plt
import torch


def side_by_side(x, y):
    """
    x and y are width by height tensors, containing values from 0 to 1
    """
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(x.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    ax1.imshow(y.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    plt.show()
