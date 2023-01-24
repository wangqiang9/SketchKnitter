import os
import math
import random
import sys

sys.stdout.write("...")
sys.stdout.flush()

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th
from tqdm import tqdm


def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        category,
        class_cond=False,
        deterministic=False,
        Nmax=96,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    classes = None
    print(f"Nmax is {Nmax}")
    dataset = SketchDataset(
        image_size,  # 96
        data_dir,
        category,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        mode="train",
        Nmax=Nmax,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class SketchDataset(Dataset):
    def __init__(
            self,
            resolution,  # 96
            image_paths,
            category,  # len(category)=10
            classes=None,  # len(classes)=10
            shard=0,  # 0
            num_shards=1,  # 1
            mode="train",
            Nmax=96,
    ):
        super().__init__()
        self.resolution = resolution

        self.sketches = None
        self.sketches_normed = None
        self.max_sketches_len = 0
        self.path = image_paths
        self.category = category
        self.mode = mode
        self.Nmax = Nmax

        tmp_sketches = []
        tmp_label = []

        for i, c in enumerate(self.category):
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            tmp_label.append([i] * len(dataset[self.mode]))
            print(f"dataset: {c} added.")

        data_sketches = np.concatenate(tmp_sketches)
        data_sketches_label = np.concatenate(tmp_label)
        data_sketches, data_sketches_label = self.purify(data_sketches,
                                                         data_sketches_label)  # data clean.  # remove toolong and too stort sketches.
        self.sketches = data_sketches.copy()
        self.sketches_label = data_sketches_label.copy()
        self.sketches_normed = self.normalize(data_sketches)

        print(f"length of trainset(normed): {len(self.sketches_normed)}")
        self.len_dataset = len(self.sketches_normed)

        # self.Nmax = self.max_size(data_sketches)  # max size of a sk  etch.

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        sketch = np.zeros((self.Nmax, 3))
        len_seq = self.sketches_normed[idx].shape[0]
        sketch[:len_seq, :] = self.sketches_normed[idx]

        out_dict = {}
        if self.sketches_label is not None:
            out_dict["y"] = np.array(self.sketches_label[idx], dtype=np.int64)

        return sketch, out_dict

    def max_size(self, sketches):
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches, labels):
        data = []
        new_labels = []
        for i, sketch in enumerate(sketches):
            # if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
            if 96 >= sketch.shape[0] > 0:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
                new_labels.append(labels[i])
        return data, new_labels

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(sketches)
        for sketch in sketches:
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][
                                                          ::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
