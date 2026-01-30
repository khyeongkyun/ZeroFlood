from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import torch


def show_rgb(img: torch.Tensor):
    img = img.detach()
    img = to_pil_image(img)
    plt.figure(figsize=(4,4))
    plt.imshow(np.asarray(img))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

def show_lulc(lulc_data: torch.Tensor, alpha=1):
    """
    Visualize LULC class values with different colors.
    
    Args:
        lulc_data: torch.Tensor of shape [256, 256] with class values 0-9
        alpha: transparency level (default 0.5)
    """
    # Define colors for classes 0-9
    colors = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', 
              '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#FFC0CB']
    labels = ['NoData',     'Water',        'Trees',    'F-Vegetation', 'Crops',
              'BuiltArea',  'BareGround',   'SnowIce',  'Clouds',       'Rangeland']
    
    lulc_array = lulc_data.cpu().numpy()
    
    # Create colormap
    cmap = ListedColormap(colors)
    
    # Create figure and plot
    plt.figure(figsize=(4,4))
    im = plt.imshow(lulc_array, cmap=cmap, vmin=0, vmax=9, alpha=alpha)
    
    # Create legend
    legend_elements = [Patch(facecolor=c, label=l) for c,l in zip(colors,labels)]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.show()