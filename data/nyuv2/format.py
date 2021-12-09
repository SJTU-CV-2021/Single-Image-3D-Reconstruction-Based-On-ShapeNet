#!/usr/bin/env python3

from pathlib import Path
from sys import path
from toolbox import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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

def format_labeled_dataset(root_path):
    labeled = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')

    n = len(labeled)

    for i in tqdm(range(n)):

        path = os.path.join(root_path, "inputs", str(i))
        color, depth, labels, instances, label_dict = labeled[i]

        w, h = color.size

        color = color.crop((1, 1, w, int(w * 530 / 730) + 1))
        depth = depth.crop((1, 1, w, int(w * 530 / 730) + 1))
        labels = depth.crop((1, 1, w, int(w * 530 / 730) + 1))
        instances = depth.crop((1, 1, w, int(w * 530 / 730) + 1))
        
        fig = plt.figure("Labeled Dataset Sample", figsize=(16, 9))

        label_dict = [obj for obj in label_dict if obj['class'] in NYU40CLASSES]

        

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

        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(os.path.join(path, 'labeled.png'))
        plt.clf()
        
        color.save(os.path.join(path, 'img.jpg'))
        with open(os.path.join(path, 'detections.json'), 'w') as f:
            f.write(str(label_dict).replace("'", '"'))
        with open(os.path.join(path, 'cam_K.txt'), 'w') as f:
            f.write("529.5   0.  365.\n0.   529.5  265. \n0.     0.     1. ")

    labeled.close()

format_labeled_dataset("format")