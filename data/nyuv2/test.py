#!/usr/bin/env python3

from pathlib import Path
from nyuv2 import *
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET_DIR = Path('dataset')

def plot_color(ax, color, title="Color"):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)

def plot_depth(ax, depth, title="Depth"):
    """Displays a depth map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    depth = depth.point(lambda x:x*0.5)
    ax.imshow(depth, cmap='nipy_spectral')

def plot_labels(ax, labels, title="Labels"):
    """Displays a labels map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(labels)

def plot_instance(ax, instance, title="Instance"):
    """Displays a instance map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(instance)

def plot_bbox(ax, color, label_dict, title="Bbox"):
    """Displays a bbox map from the NYU dataset."""
    w, h = color.size
    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)
    for item in label_dict:
        ax.plot([item['bbox'][0], item['bbox'][2], item['bbox'][2], item['bbox'][0], item['bbox'][0]],  # col
         [item['bbox'][1], item['bbox'][1], item['bbox'][3], item['bbox'][3], item['bbox'][1]],  # row
         color='red', marker='.', ms=0)
        ax.text(item['bbox'][0], item['bbox'][1], item['class'])


def test_labeled_dataset(path):
    labeled = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')

    color, depth, labels, instances, label_dict = labeled[22]

    NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
    
    fig = plt.figure("Labeled Dataset Sample", figsize=(16, 9))

    label_dict = [obj for obj in label_dict if obj['class'] in NYU40CLASSES]

    w, h = color.size

    for item in label_dict:
        item['bbox'][0] = w - item['bbox'][0]
        item['bbox'][2] = w - item['bbox'][2]
        item['bbox'][2], item['bbox'][0] = item['bbox'][0], item['bbox'][2]

    ax = fig.add_subplot(2, 2, 1)
    plot_color(ax, color)

    ax = fig.add_subplot(2, 2, 2)
    plot_bbox(ax, color, label_dict)
    
    ax = fig.add_subplot(2, 2, 3)
    plot_labels(ax, labels)

    ax = fig.add_subplot(2, 2, 4)
    plot_instance(ax, instances)

    plt.savefig('test_labeled_dataset.png')
    plt.show()

    
    if not os.path.exists(path):
        os.makedirs(path)
        
    color.save(os.path.join(path, 'img.jpg'))
    with open(os.path.join(path, 'detections.json'), 'w') as f:
        f.write(str(label_dict).replace("'", '"'))
    with open(os.path.join(path, 'cam_K.txt'), 'w') as f:
        f.write("529.5   0.  365.\n0.   529.5  265. \n0.     0.     1. ")
    labeled.close()

def test_raw_dataset():
    # Pick the first raw dataset part we find
    raw_archive_path = next(DATASET_DIR.glob('*.zip'))

    raw_archive = RawDatasetArchive(raw_archive_path)
    frame = raw_archive[4]
    depth_path, color_path = Path('dataset') / frame[0], Path('dataset') / frame[1]

    if not (depth_path.exists() and color_path.exists()):
        raw_archive.extract_frame(frame)

    color = load_color_image(color_path)
    depth = load_depth_image(depth_path)

    fig = plt.figure("Raw Dataset Sample", figsize=(12, 5))

    before_proj_overlay = color_depth_overlay(color, depth, relative=True)

    ax = fig.add_subplot(1, 2, 1)
    plot_color(ax, before_proj_overlay, "Before Projection")

    # TODO: project depth and RGB image
    plt.savefig('test_raw_dataset.png')
    plt.show()

test_labeled_dataset("../Total3DUnderstanding/demo/inputs/6")
# test_raw_dataset()
