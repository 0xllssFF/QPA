from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import config
import logging
import torch
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL

def get_logger():
    logger = logging.getLogger('normal')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s %(pathname)s:%(lineno)d %(levelname)s]: %(message)s','%H:%M:%S,%m/%d')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger    

# get center crop
# def load_image(path):
#     image = PIL.Image.open(path)
#     if image.height > image.width:
#         height_off = int((image.height - image.width)/2)
#         image = image.crop((0, height_off, image.width, height_off+image.width))
#     elif image.width > image.height:
#         width_off = int((image.width - image.height)/2)
#         image = image.crop((width_off, 0, width_off+image.height, image.height))
#     image = image.resize((299, 299))
#     img = np.asarray(image).astype(np.float32) / 255.0
#     if img.ndim == 2:
#         img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
#     if img.shape[2] == 4:
#         # alpha channel
#         img = img[:,:,:3]
#     return img

def load_image(path, transform):
    img = transform(PIL.Image.open(path).convert("RGB"))
    img = np.asarray(img).astype(np.float32)
    return img

def load_image_BL(path, transform):
    img = transform(PIL.Image.open(path))
    img = np.asarray(img).astype(np.float32)
    return img 
    
def get_imagenet_list(index, transform, imagenet_path=None, mode = None):
    if mode == 'train':
        data_path = os.path.join(imagenet_path, 'train')
        data_path_img = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        image_paths = []
        for path in data_path_img:
            image_paths.extend([os.path.join(path, i) for i in os.listdir(path)])
        image_paths = sorted(image_paths)
        # assert len(image_paths) == 50000
        labels_path = os.path.join(imagenet_path, 'train.txt')
        with open(labels_path) as labels_file:
            labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
            labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    elif mode == 'test':
        data_path = os.path.join(imagenet_path, 'val')
        data_path_img = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        image_paths = []
        for path in data_path_img:
            image_paths.extend([os.path.join(path, i) for i in os.listdir(path)])
        image_paths = sorted(image_paths)
        assert len(image_paths) == 50000
        labels_path = os.path.join(imagenet_path, 'val.txt')
        with open(labels_path) as labels_file:
            labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
            labels = {os.path.basename(i[0]): int(i[1]) for i in labels}
    def get(index):
        all_x = []
        all_y = []
        for i in index:
            path = image_paths[i]
            x = load_image(path, transform)
            y = labels[os.path.basename(path)]
            all_x.append(x)
            all_y.append(y)
        return all_x, all_y
    return get(index)

def get_cifar10_list(index, transform, cifar10_path=None):
    image_paths = [os.path.join(cifar10_path, i) for i in os.listdir(cifar10_path)]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # assert len(image_paths) == 50000

    def get(index):
        all_x = []
        for i in index:
            path = image_paths[i]
            x = load_image_BL(path, transform)
            all_x.append(x)
        return all_x
    return get(index)

def get_mnist_list(index, transform, mnist_path=None):
    image_paths = [os.path.join(mnist_path, i) for i in os.listdir(mnist_path)]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    def get(index):
        all_x = []
        for i in index:
            path = image_paths[i]
            x = load_image_BL(path, transform)
            all_x.append(x)
        return all_x
    return get(index)

def get_celebahq_list(index, transform, celebahq_path=None):
    image_paths = [os.path.join(celebahq_path, i) for i in os.listdir(celebahq_path)]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    def get(index):
        all_x = []
        for i in index:
            path = image_paths[i]
            x = load_image(path, transform)
            all_x.append(x)
        return all_x
    return get(index)

def get_image(index, imagenet_path=None, mode = None):
    if mode == 'train':
        data_path = os.path.join(imagenet_path, 'train')
        data_path_img = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        image_paths = []
        for path in data_path_img:
            image_paths.extend([os.path.join(path, i) for i in os.listdir(path)])
        image_paths = sorted(image_paths)
        # assert len(image_paths) == 50000
        labels_path = os.path.join(imagenet_path, 'train.txt')
        with open(labels_path) as labels_file:
            labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
            labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    elif mode == 'test':
        data_path = os.path.join(imagenet_path, 'val')
        data_path_img = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        image_paths = []
        for path in data_path_img:
            image_paths.extend([os.path.join(path, i) for i in os.listdir(path)])
        image_paths = sorted(image_paths)
        assert len(image_paths) == 50000
        labels_path = os.path.join(imagenet_path, 'val.txt')
        with open(labels_path) as labels_file:
            labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
            labels = {os.path.basename(i[0]): int(i[1]) for i in labels}
    def get(index):
        path = image_paths[index]
        x = load_image(path)
        y = labels[os.path.basename(path)]
        return x,y
    return get(index)