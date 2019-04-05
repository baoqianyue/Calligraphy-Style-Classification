import numpy as np
import os
from PIL import Image


def load_data(dir):
    data = []
    labels = []
    for img in os.listdir(dir):
        if img.split("_")[0] == '0':
            labels.append([1, 0, 0, 0])
            img_path = os.path.join(dir, img)
            src = Image.open(img_path)
            arr = np.asarray(src, dtype=np.float32)
            data.append(arr)
        elif img.split("_")[0] == '1':
            labels.append([0, 1, 0, 0])
            img_path = os.path.join(dir, img)
            src = Image.open(img_path)
            arr = np.asarray(src, dtype=np.float32)
            data.append(arr)
        elif img.split("_")[0] == '2':
            labels.append([0, 0, 1, 0])
            img_path = os.path.join(dir, img)
            src = Image.open(img_path)
            arr = np.asarray(src, dtype=np.float32)
            data.append(arr)
        else:
            labels.append([0, 0, 0, 1])
            img_path = os.path.join(dir, img)
            src = Image.open(img_path)
            arr = np.asarray(src, dtype=np.float32)
            data.append(arr)
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels


def random_read_batch(data, labels, batch_size):
    """shuffle 选取batch"""
    nums = data.shape[0]
    range_batch = np.random.randint(0, nums, [batch_size])
    batch = data[range_batch, :]
    label = labels[range_batch, :]
    return batch, label



