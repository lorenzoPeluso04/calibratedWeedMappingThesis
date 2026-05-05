import itertools
import os
import torch
import torchvision

from torch.utils.data import Dataset


class WeedMapDataset(Dataset):
    id2class = {
        0: "background",
        1: "crop",
        2: "weed",
    }

    # Inizialize the object
    def __init__(
        self,
        root,
        channels,
        fields,
        gt_folder=None,
        transform=None,
        target_transform=None,
        return_path=False,
        return_ndvi=False, # Return NDVI as extra channel
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.fields = fields
        self.return_ndvi = return_ndvi
        self.channels = channels

        # Create a Dict (index) where there is the image name for each image (a patch of the orthomosaic maps)
        # Es: index[0] = {'000': '0.png'}
        if gt_folder is None:
            self.gt_folders = {
                field: os.path.join(self.root, field, "groundtruth")
                for field in self.fields
            }
        else:
            self.gt_folders = {
                field: os.path.join(gt_folder, field) for field in self.fields
            }
            for k, v in self.gt_folders.items():
                if os.path.isdir(os.path.join(v, os.listdir(v)[0])):
                    self.gt_folders[k] = os.path.join(v, "groundtruth")

        self.index = [
            (field, filename) for field in self.fields for filename in os.listdir(self.gt_folders[field])
        ]

    # Return the number of images
    def __len__(self):
        return len(self.index)

    # Return the specific ground-truth image
    def _get_gt(self, gt_path):
        gt = torchvision.io.read_image(gt_path)
        gt = gt[[2, 1, 0], ::]
        gt = gt.argmax(dim=0)
        gt = self.target_transform(gt)
        return gt

    # Returns a specific image (as a Tensor) by concatenating all the image channels
    def _get_image(self, field, filename):
        channels = []
        for channel_folder in self.channels:
            channel_path = os.path.join(
                self.root,
                field,
                channel_folder,
                filename
            )
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return self.transform(channels)

    def _get_ndvi(self, field, filename):
        nir_red_path = [
            os.path.join(
                self.root,
                field,
                ch,
                filename
            ) for ch in ["NIR", "R"]
        ]
        nir_red = [torchvision.io.read_image(channel_path).float() for channel_path in nir_red_path]
        ndvi = (nir_red[0] - nir_red[1]) / (nir_red[0] + nir_red[1])
        # Replaces NaN values with 0
        ndvi[torch.isnan(ndvi)] = 0
        return ndvi

    # Return a dict that contains the ith image of the dataset and its ground-truth
    def __getitem__(self, i):
        field, filename = self.index[i]
        gt_path = os.path.join(
            self.gt_folders[field], filename
        )
        gt = self._get_gt(gt_path)
        channels = self._get_image(field, filename)

        data_dict = {
            "image": channels,
            "target": gt,
        }
        if self.return_path:
            data_dict["name"] = gt_path

        if self.return_ndvi:
            ndvi = self._get_ndvi(field, filename)
            data_dict.ndvi = ndvi

        return data_dict