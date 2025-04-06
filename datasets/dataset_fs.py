import os
import random
import torch
from torch.utils.data import Dataset
from datasets.data_process import load3DImages2Tensor
from datasets.data_patchify import split_image, reconstruct_image


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Input:
    - batch (list): List of tuples containing patches and their indices.

    Output:
    - tuple: Stacked patches and their corresponding indices.
    """
    patches = []
    image_indices = []
    z_indices = []
    w_indices = []
    h_indices = []
    for item in batch:
        patch, image_idx, z_idx, w_idx, h_idx = item
        patches.append(patch)
        image_indices.append(image_idx)
        z_indices.append(z_idx)
        w_indices.append(w_idx)
        h_indices.append(h_idx)
    patches = torch.stack(patches)
    return (
        patches,
        torch.tensor(image_indices),
        torch.tensor(z_indices),
        torch.tensor(w_indices),
        torch.tensor(h_indices),
    )


class ReadDatasets(Dataset):
    """
    Dataset class for reading and processing 3D image data.

    Input:
    - dataPath (str): Path to the dataset.
    - mode (str): Mode of operation ('train', 'val', 'test').
    - dataType (str): Type of data ('2D', '3D', '4D').
    - dataExtension (str): File extension of the data (e.g., 'tif').
    - z_patch, w_patch, h_patch (int): Patch sizes in z, w, and h dimensions.
    - z_overlap, w_overlap, h_overlap (float): Overlap ratios for patches.
    - patch_num (int): Maximum number of patches to extract. Default: -1 (no limit).
    - dataNum (int): Number of data samples to load. Default: -1 (load all).

    Output:
    - torch.Tensor: Processed dataset.
    """

    def __init__(
        self,
        dataPath: str,
        mode: str,
        dataType: str,
        dataExtension: str,
        z_patch=2,
        w_patch=150,
        h_patch=150,
        z_overlap=0.5,
        w_overlap=0.5,
        h_overlap=0.5,
        patch_num=-1,
        dataNum=-1,
    ) -> torch.Tensor:
        super(ReadDatasets, self).__init__()
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be types in 'train','val','test'!"
        assert dataType in ["2D", "3D", "4D"], "mode must be types in '2D','3D','4D'!"

        self.dataPath = dataPath
        self.dataType = dataType
        self.dataNum = dataNum
        self.mode = mode
        self.z_patch, self.w_patch, self.h_patch = z_patch, w_patch, h_patch
        self.z_overlap, self.w_overlap, self.h_overlap = z_overlap, w_overlap, h_overlap
        self.patch_num = patch_num
        self.dataExtension = dataExtension
        if self.dataType == "3D":
            if self.dataExtension == "tif":
                self.inputFileNames = [
                    f
                    for f in os.listdir(self.dataPath)
                    if f.endswith(f".{self.dataExtension}")
                ]
                self.inputsNum = len(self.inputFileNames)
                if self.mode == "train" or self.mode == "test" or self.mode == "val":
                    self.imageAll = load3DImages2Tensor(
                        dataPath=self.dataPath,
                        dataExtension=self.dataExtension,
                        dataNum=self.dataNum,
                    )
                    self.original_shapes = [image.shape for image in self.imageAll]
                    # Initialize lists to store mean and std values for each image
                    self.image_stds = []
                    self.image_means = []
                    # Iterate through each image in self.imageAll
                    for i, image in enumerate(self.imageAll):
                        
                        std_value = image.std()
                        mean_value = image.mean()
                        # Save mean and std values
                        self.image_stds.append(std_value)
                        self.image_means.append(mean_value)
                        # Normalize the image
                        self.imageAll[i] =  (image - mean_value)/std_value
                    # Split the images and get indices
                    self.indices, self.paddings = self.split_dataset(
                        self.imageAll,
                        self.z_patch,
                        self.w_patch,
                        self.h_patch,
                        self.z_overlap,
                        self.w_overlap,
                        self.h_overlap,
                        self.patch_num
                    )
                    # Print the total number of patches
                    print(f"Total patches: {len(self.indices)}")

    def split_dataset(
            self, images, z_patch, w_patch, h_patch, z_overlap, w_overlap, h_overlap, patch_num
    ):
        """
        Splits the dataset into patches.

        Input:
        - images (list): List of 3D image tensors.
        - z_patch, w_patch, h_patch (int): Patch sizes in z, w, and h dimensions.
        - z_overlap, w_overlap, h_overlap (float): Overlap ratios for patches.
        - patch_num (int): Maximum number of patches to extract. Default: -1 (no limit).

        Output:
        - indices (list): List of patch indices.
        - paddings (list): List of padding values for each image.
        """
        indices = []
        paddings = []
        for image_idx, image in enumerate(images):
            # Calculate the number of patches in width and height dimensions
            w_steps = int((image.shape[1] - w_patch) / (w_patch * (1 - w_overlap))) + 1
            h_steps = int((image.shape[2] - h_patch) / (h_patch * (1 - h_overlap))) + 1
            patches_per_image = w_steps * h_steps

            # Adjust z dimension splitting if patch_num is specified
            if patch_num != -1:
                # Calculate the required number of z dimension patches
                z_steps = max(1, (patch_num + patches_per_image - 1) // patches_per_image)
                # Dynamically adjust z_patch and z_overlap
                z_patch = max(1, image.shape[0] // z_steps)
                z_overlap = max(0,
                                (image.shape[0] - z_patch * z_steps) / (z_patch * (z_steps - 1))) if z_steps > 1 else 0

            # Split the image
            image_indices, padding = split_image(
                image, z_patch, w_patch, h_patch, z_overlap, w_overlap, h_overlap
            )
            print("z_patch:",z_patch,"  z_overlap:",z_overlap)
            indices.extend(
                [
                    (image_idx, z_idx, w_idx, h_idx)
                    for (_, z_idx, w_idx, h_idx) in image_indices
                ]
            )
            paddings.append(padding)

            # Truncate indices if patch_num is specified and exceeded
            if patch_num != -1 and len(indices) >= patch_num:
                indices = indices[:patch_num]
                break

        return indices, paddings

    def reconstruct_dataset(self, patches, indices):
        """
        Reconstructs the dataset from patches.

        Input:
        - patches (torch.Tensor): Tensor containing patches.
        - indices (list): List of patch indices.

        Output:
        - list: List of reconstructed images.
        """
        reconstructed_images = reconstruct_image(
            patches,
            indices,
            self.original_shapes,
            self.z_patch,
            self.w_patch,
            self.h_patch,
            self.z_overlap,
            self.w_overlap,
            self.h_overlap,
            self.paddings,
        )
        for i in range(len(reconstructed_images)):
            std_value = self.image_stds[i]
            mean_value = self.image_means[i]
            reconstructed_images[i] = reconstructed_images[i] * std_value + mean_value
        return reconstructed_images

    def __getitem__(self, item):
        """
        Retrieves a single patch and its indices.

        Input:
        - item (int): Index of the patch to retrieve.

        Output:
        - tuple: Patch tensor and its corresponding indices.
        """
        image_idx, z_idx, w_idx, h_idx = self.indices[item]
        image = self.imageAll[image_idx]
        z_pad, w_pad, h_pad = self.paddings[image_idx]
        padded_image = torch.nn.functional.pad(
            image, (0, h_pad, 0, w_pad, 0, z_pad), mode="constant", value=0
        )
        patch = padded_image[
            z_idx : z_idx + self.z_patch,
            w_idx : w_idx + self.w_patch,
            h_idx : h_idx + self.h_patch,
        ]

        return torch.unsqueeze(patch, dim=0), image_idx, z_idx, w_idx, h_idx


    def __len__(self):
        """
        Returns the total number of patches.

        Input:
        - None

        Output:
        - int: Total number of patches.
        """
        return len(self.indices)
