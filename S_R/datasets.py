# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png") or filename.endswith("_100000.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}').replace('\\', '/')

def image_path_city(root, name):
    return os.path.join(root, f'{name}').replace('\\', '/')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='val', target = False):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset).replace('\\', '/')
        self.labels_root = os.path.join(root, 'gtFine/' + subset).replace('\\', '/')
        self.target = target

        print (self.images_root)
        self.filenames = [os.path.join(dp, f).replace('\\', '/') for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        # if (set == 'train'):
        self.filenamesGt = [os.path.join(dp, f).replace('\\', '/') for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        # else:
        #     self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]
        self.filenamesGt.sort()
        print(len(self.filenames))
        print(len(self.filenamesGt))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        #print(filename)

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')

        
        if self.input_transform is not None:
            image = self.input_transform(image)

        if (not self.target):
            filenameGt = self.filenamesGt[index]
            with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
                label = load_image(f).convert('P')

            if self.target_transform is not None:
                label = self.target_transform(label)

            return image, label, np.array(image.size()), filename
        
        return image, np.array(image.size()), filename

    def __len__(self):
        return len(self.filenames)

