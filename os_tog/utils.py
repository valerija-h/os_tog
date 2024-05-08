import matplotlib.pyplot as plt
import random
import colorsys
import torch
import torchvision
import numpy as np
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from matplotlib.transforms import Affine2D

def visualize(image, boxes=None, masks=None, class_ids=None, grasps=None, figsize=(6, 4), ax=None, title=""):
    """ Function for plotting images, bounding boxes, masks and grasps. """
    # get number of instances
    N = 0
    if boxes is not None:
        N = boxes.shape[0]
    elif masks is not None:
        N = masks.shape[0]
    elif grasps is not None:
        N = grasps.shape[0]
    
    # create figure
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    if N > 0:
        # generate random colours
        hsv = [(i / N, 1, 1.0) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)

    ax.axis('off')
    ax.set_title(title)
    
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i] # get colour for current instance

        # plot bounding boxes
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # plot masks
        if masks is not None:
            mask = masks[i, :, :]
            masked_image = apply_mask(masked_image, mask, color)
        
        # plot grasps
        if grasps is not None:
            x, y, w, h, t = grasps[i]
            w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
                    h / 2) * np.cos(t)
            bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
            br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
            ax.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
            ax.plot([br_x, tr_x], [br_y, tr_y], c='black')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor=color, facecolor='none',
                                    transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
            ax.add_patch(rect)
    ax.imshow(masked_image.astype(np.uint8))

def visualize_affordances(image, aff_labels=None, aff_labels_names=None, aff_masks=None, ax=None, figsize=(12, 8), title=""):
    """ Function for plotting affordances. """
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    A = 0
    if aff_labels is not None:
        A = aff_labels.shape[0]
        colors = generate_colors(aff_labels_names.shape[0])
    
    masked_image = image.astype(np.uint32).copy()    
    for i in range(A): # for each affordance
        # plot masks
        if aff_masks is not None:
            mask = aff_masks[i, :, :]
            aff_label_idx = aff_labels[i]
            masked_image = apply_mask(masked_image, mask, colors[aff_label_idx-1], alpha=0.7)

    if aff_labels_names is not None:
        for i in range(aff_labels_names.shape[0]):
            ax.plot([], [], color=colors[i], label=aff_labels_names[i])
        
    ax.legend(prop={'size': 7})
    ax.axis('off')
    ax.set_title(title)
    ax.imshow(masked_image.astype(np.uint8))

def plot_triplet(anchor, positive, negative):  
    def feature_to_img(feature):
        return np.reshape(feature, (-1, 64)) 

    def prepare_img(img):
        if torch.is_tensor(img):
            img = torchvision.transforms.ToPILImage()(img)
        return img

    def show(ax, image, sub_title=''):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if sub_title != '':
            ax.set_title(sub_title)
    
    if anchor.ndim == 1: # if it's a feature, reshape into image
        anchor = feature_to_img(anchor)
    
    fig = plt.figure(figsize=(9, 9))
    axs = fig.subplots(1, 3)
    show(axs[0], prepare_img(anchor), "anchor")
    show(axs[1], prepare_img(positive), "positive")
    show(axs[2], prepare_img(negative), "negative")


def apply_mask(image, mask, color, alpha=0.5):
    """Apply a binary mask to an image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image    

def print_losses(loss_hist, epoch, prefix=''):
    """ Prints out the training/validation losses from a given dictionary. """
    line = f'Epoch {epoch}'
    for name, value in loss_hist[epoch].items():
        line += f' - {prefix}{name}: {sum(value)/len(value):.4f}'
    print(line)

def generate_colors(N):
    "@param N (int) - number of colours to generate"
    # generate random colours
    hsv = [(i / N, 1, 1.0) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors

def get_points(bbox, t):
    """ Convert grasp from [xmin, ymin, xmax, ymax] and 't' value format to ((tl_x, tl_y), (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y)). 
    Note that t is the theta value. """
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    x, y = xmax - (w / 2), ymax - (h / 2)

    w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
            h / 2) * np.cos(t)
    bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
    br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
    return (tl_x, tl_y), (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y)