import gzip
import os
import shutil
import urllib.request

import numpy as np

DATASET_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_if_needed(url, destination):
    if os.path.exists(destination):
        return destination

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 16:
        raise ValueError(f"MNIST image file is too small: {path}")

    magic = int.from_bytes(raw[:4], "big")
    if magic != 2051:
        raise ValueError(f"Unexpected MNIST image magic number in {path}: {magic}")

    num_images = int.from_bytes(raw[4:8], "big")
    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")

    image_data = np.frombuffer(raw, dtype=np.uint8, count=num_images * rows * cols, offset=16)
    return image_data.reshape(num_images, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 8:
        raise ValueError(f"MNIST label file is too small: {path}")

    magic = int.from_bytes(raw[:4], "big")
    if magic != 2049:
        raise ValueError(f"Unexpected MNIST label magic number in {path}: {magic}")

    count = int.from_bytes(raw[4:8], "big")
    labels = np.frombuffer(raw, dtype=np.uint8, count=count, offset=8).astype(np.float32)
    return labels


base_dir = os.path.dirname(os.path.abspath(__file__))

train_images_path = os.path.join(base_dir, "train-images-idx3-ubyte.gz")
train_labels_path = os.path.join(base_dir, "train-labels-idx1-ubyte.gz")
test_images_path = os.path.join(base_dir, "t10k-images-idx3-ubyte.gz")
test_labels_path = os.path.join(base_dir, "t10k-labels-idx1-ubyte.gz")

download_if_needed(DATASET_URLS["train_images"], train_images_path)
download_if_needed(DATASET_URLS["train_labels"], train_labels_path)
download_if_needed(DATASET_URLS["test_images"], test_images_path)
download_if_needed(DATASET_URLS["test_labels"], test_labels_path)

train_images = read_idx_images(train_images_path)
train_labels = read_idx_labels(train_labels_path)
test_images = read_idx_images(test_images_path)
test_labels = read_idx_labels(test_labels_path)

train_images.tofile(os.path.join(base_dir, "train_images.mat"))
train_labels.tofile(os.path.join(base_dir, "train_labels.mat"))
test_images.tofile(os.path.join(base_dir, "test_images.mat"))
test_labels.tofile(os.path.join(base_dir, "test_labels.mat"))

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)