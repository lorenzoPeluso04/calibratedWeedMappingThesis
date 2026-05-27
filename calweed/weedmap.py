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
                if os.path.isdir(os.path.join(v, "groundtruth")):
                    self.gt_folders[k] = os.path.join(v, "groundtruth")

        self.index = []
        for field in self.fields:
            gt_folder = self.gt_folders[field]
            if not os.path.isdir(gt_folder):
                print(f"Warning: Ground truth folder not found: {gt_folder}")
                continue
            for filename in os.listdir(gt_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.index.append((field, filename))

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
        print(f"Loading image: {gt_path}")
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

    # Calculate weed percentage for a single image
    def get_weed_percentage(self, idx):
        """
        Calculate the percentage of weed pixels in a single image.
        
        Args:
            idx: Index of the image in the dataset
            
        Returns:
            float: Percentage of weed coverage (0-100)
        """
        field, filename = self.index[idx]
        gt_path = os.path.join(self.gt_folders[field], filename)
        gt = self._get_gt(gt_path)
        
        # Count weed pixels (class 2)
        weed_pixels = (gt == 2).sum().item()
        total_pixels = gt.numel()
        
        weed_percentage = (weed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        return weed_percentage

    # Calculate weed statistics for the entire dataset
    def get_dataset_weed_statistics(self):
        """
        Calculate weed coverage statistics for the entire dataset.
        
        Returns:
            dict: Statistics including total weed percentage, per-field percentages,
                  and pixel counts for each class
        """
        total_weed_pixels = 0
        total_crop_pixels = 0
        total_bg_pixels = 0
        
        field_stats = {field: {"weed": 0, "crop": 0, "background": 0, "total": 0} 
                       for field in self.fields}
        
        for idx in range(len(self)):
            field, filename = self.index[idx]
            gt_path = os.path.join(self.gt_folders[field], filename)
            gt = self._get_gt(gt_path)
            
            # Count pixels per class
            weed_pixels = (gt == 2).sum().item()
            crop_pixels = (gt == 1).sum().item()
            bg_pixels = (gt == 0).sum().item()
            total_pixels = gt.numel()
            
            total_weed_pixels += weed_pixels
            total_crop_pixels += crop_pixels
            total_bg_pixels += bg_pixels
            
            # Update field statistics
            field_stats[field]["weed"] += weed_pixels
            field_stats[field]["crop"] += crop_pixels
            field_stats[field]["background"] += bg_pixels
            field_stats[field]["total"] += total_pixels
        
        total_pixels = total_weed_pixels + total_crop_pixels + total_bg_pixels
        
        # Calculate percentages
        stats = {
            "total_weed_percentage": (total_weed_pixels / total_pixels * 100) if total_pixels > 0 else 0,
            "total_crop_percentage": (total_crop_pixels / total_pixels * 100) if total_pixels > 0 else 0,
            "total_bg_percentage": (total_bg_pixels / total_pixels * 100) if total_pixels > 0 else 0,
            "total_pixels": total_pixels,
            "weed_pixels": total_weed_pixels,
            "crop_pixels": total_crop_pixels,
            "bg_pixels": total_bg_pixels,
            "field_statistics": {}
        }
        
        # Calculate per-field percentages
        for field, counts in field_stats.items():
            total = counts["total"]
            stats["field_statistics"][field] = {
                "weed_percentage": (counts["weed"] / total * 100) if total > 0 else 0,
                "crop_percentage": (counts["crop"] / total * 100) if total > 0 else 0,
                "bg_percentage": (counts["background"] / total * 100) if total > 0 else 0,
                "total_pixels": total,
                "weed_pixels": counts["weed"],
                "crop_pixels": counts["crop"],
                "bg_pixels": counts["background"],
            }
        
        return stats
    

if __name__ == "__main__":
    # Example usage
    dataset = WeedMapDataset(
        root="RoWeeder/dataset/patches/512",
        channels=["RGB", "NIR"],
        fields=["000", "001"],  # Update with your actual field names
        gt_folder=None,  # Set to None to use default groundtruth folders, or specify custom path
        transform=torchvision.transforms.ToTensor(),
        target_transform=lambda x: x,  # No transformation for target
        return_path=True,
        return_ndvi=True
    )
    
    if len(dataset) == 0:
        print("Error: No images found in the dataset. Please check your paths.")
    else:
        # Get weed percentage for the first image
        weed_percentage = dataset.get_weed_percentage(0)
        print(f"Weed percentage in the first image: {weed_percentage:.2f}%")
        
        # Get overall dataset statistics
        stats = dataset.get_dataset_weed_statistics()
        print("\nDataset Weed Statistics:")
        print(f"Total weed percentage: {stats['total_weed_percentage']:.2f}%")
        print(f"Total crop percentage: {stats['total_crop_percentage']:.2f}%")
        print(f"Total background percentage: {stats['total_bg_percentage']:.2f}%")
        print(f"\nPer-field statistics:")
        for field, field_stats in stats['field_statistics'].items():
            print(f"\n  {field}:")
            print(f"    Weed: {field_stats['weed_percentage']:.2f}%")
            print(f"    Crop: {field_stats['crop_percentage']:.2f}%")
            print(f"    Background: {field_stats['bg_percentage']:.2f}%")