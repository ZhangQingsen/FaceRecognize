import os

import numpy as np
import cv2
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


class DataLoader(Dataset):
    def __init__(self, df, input_shape):
        self.input_shape = input_shape
        self.df = df
        self.length = df.shape[0]
        self.num_classes = df["Label"].max() + 1

        # ------------------------------------#
        #   路径和标签
        # ------------------------------------#
        self.paths = []
        self.labels = []

        self.load_dataset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # ------------------------------------#
        #   创建全为零的矩阵
        # ------------------------------------#
        images = np.zeros((3, 3, self.input_shape, self.input_shape))
        labels = np.zeros(3)

        # ------------------------------#
        #   先获得两张同一个人的人脸
        #   用来作为anchor和positive
        # ------------------------------#
        c = np.random.randint(0, self.num_classes - 1)
        selected_path = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c = np.random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]

        # ------------------------------------#
        #   随机选择两张
        # ------------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)
        # ------------------------------------#
        #   打开图片并放入矩阵
        # ------------------------------------#
        image = Image.open(selected_path[image_indexes[0]]).resize((self.input_shape, self.input_shape),
                                                                   Image.ANTIALIAS)
        image = resize_image(image, [self.input_shape, self.input_shape], letterbox_image=True)
        image = (np.array(image, dtype='float32')) / 255.0

        image = np.transpose(image, [2, 0, 1])

        images[0, :, :, :] = image
        labels[0] = c

        image = Image.open(selected_path[image_indexes[1]]).resize((self.input_shape, self.input_shape),
                                                                   Image.ANTIALIAS)

        image = resize_image(image, [self.input_shape, self.input_shape], letterbox_image=True)
        image = (np.array(image, dtype='float32')) / 255.0

        image = np.transpose(image, [2, 0, 1])
        images[1, :, :, :] = image
        labels[1] = c

        # ------------------------------#
        #   取出另外一个人的人脸
        # ------------------------------#
        different_c = list(range(self.num_classes))
        different_c.pop(c)
        different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
        current_c = different_c[different_c_index[0]]
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]

        # ------------------------------#
        #   随机选择一张
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        image = Image.open(selected_path[image_indexes[0]]).resize((self.input_shape, self.input_shape),
                                                                   Image.ANTIALIAS)

        image = resize_image(image, [self.input_shape, self.input_shape], letterbox_image=True)
        image = (np.array(image, dtype='float32')) / 255.0

        image = np.transpose(image, [2, 0, 1])
        images[2, :, :, :] = image
        labels[2] = current_c
        print("======>",images.shape,labels.shape)
        return images, labels

    def load_dataset(self):
        self.paths = np.array(self.df["ImagePath"], dtype=np.object)
        self.labels = np.array(self.df["Label"])

    # DataLoader中collate_fn使用


def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    return images, labels
