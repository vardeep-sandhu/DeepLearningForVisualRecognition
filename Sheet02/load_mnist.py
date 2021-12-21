
# -*- coding: utf-8 -*-
import os
import gzip
import torch as th
import numpy as np

def load_mnist(dataset="training", path="."):
    if dataset not in  ['training', 'testing']:
        raise ValueError("dataset has to be either 'training' or 'testing'")

    if dataset == 'training':
        kind = 'train'
    else:
        kind = 't10k'

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).astype(np.float32)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28).astype(np.float32)

    return th.from_numpy(images), th.from_numpy(labels)
    
    
    