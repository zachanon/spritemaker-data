import json
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
import numpy as np
from PIL import Image


def collate_sprites(sample):
    """
    Zero pads the batch of sprite images to a uniform shape
    """

    ims, labels = zip(*sample)

    max_height = 0
    max_width = 0
    for im in ims:
        height, width = im.shape[0], im.shape[1]

        max_height = height if height > max_height else max_height
        max_width = width if width > max_width else max_width

    padded = []
    for im in ims:

        diff_h = max_height - im.shape[0]
        diff_w = max_width - im.shape[1]

        padded.append(pad(im, (0, 0, 0, diff_w, 0, diff_h)))

    return [(ims[i], labels[i]) for i in range(len(ims))]


class SpritesDataset(Dataset):
    def __init__(self, labels_fp, data_fp):
        """
        Args:
            labels_fp (string): path to the json file deliminating image tags
            data_fp (string): path to the directory holding the images
        """

        # Save the filepath data for loading
        self.filepath = data_fp

        # Build a dictionary of labels for each image
        with open(labels_fp) as file:
            labels = json.load(file)

        # Build a list of file titles (for calling a specific index)
        files = []
        for key in labels.keys():
            files.append(key)

        self.labels = labels
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.filepath + "/" + self.files[idx] + ".jpg")
        image = np.array(image, dtype=np.float32)
        label = self.labels[self.files[idx]]

        return torch.tensor(image, dtype=torch.float), label
