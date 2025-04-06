import os
import torch
import h5py
from torch.utils.data import Dataset
from datasets.data_process import load3DImages2Tensor
from datasets.data_patchify import split_image, reconstruct_image
from tqdm import tqdm


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
    Dataset class for reading and processing 3D image data with HDF5 storage.

    Input:
    - dataPath (str): Path to the dataset.
    - mode (str): Mode of operation ('train', 'val', 'test').
    - dataType (str): Type of data ('2D', '3D', '4D').
    - dataExtension (str): File extension of the data (e.g., 'tif').
    - z_patch, w_patch, h_patch (int): Patch sizes in z, w, and h dimensions.
    - z_overlap, w_overlap, h_overlap (float): Overlap ratios for patches.
    - patch_num (int): Maximum number of patches to extract. Default: -1 (no limit).
    - dataNum (int): Number of data samples to load. Default: -1 (load all).
    - h5_file_path (str): Path to the temporary HDF5 file for storing patches.

    Output:
    - torch.Tensor: Processed dataset.
    """

    def __init__(
        self,
        dataPath: str,
        mode: str,
        dataType: str,
        dataExtension: str,
        z_patch=224,
        w_patch=224,
        h_patch=224,
        z_overlap=0.1,
        w_overlap=0.1,
        h_overlap=0.1,
        patch_num=-1,
        dataNum=-1,
        h5_file_path="patches.h5"
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
        self.h5_file_path = f"{dataPath}/{h5_file_path}"

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
                        self.imageAll[i] = (image - mean_value) / std_value
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
                    # Output the total number of patches
                    print(f"Total patches: {len(self.indices)}")
                    # Save the patches to an HDF5 file
                    self.save_patches_to_h5()

        # Open the HDF5 file
        try:
            self.h5_file = h5py.File(self.h5_file_path, 'r')
        except Exception as e:
            self.h5_file.close()
            os.remove(self.h5_file_path)
            self.h5_file = h5py.File(self.h5_file_path, 'r')

    def __del__(self):
        """
        Destructor to close and remove the HDF5 file.

        Input:
        - None

        Output:
        - None
        """
        self.h5_file.close()
        os.remove(self.h5_file_path)
        print(f"File {self.h5_file_path} has been closed and removed.")

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
            # Calculate the number of patches in w and h dimensions
            w_steps = int((image.shape[1] - w_patch) / (w_patch * (1 - w_overlap))) + 1
            h_steps = int((image.shape[2] - h_patch) / (h_patch * (1 - h_overlap))) + 1
            patches_per_image = w_steps * h_steps

            # Adjust z dimension splitting if patch_num is specified
            if patch_num != -1:
                # Calculate the required number of patches in z dimension
                z_steps = max(1, (patch_num + patches_per_image - 1) // patches_per_image)
                # Dynamically adjust z_patch and z_overlap
                z_patch = max(1, image.shape[0] // z_steps)
                z_overlap = max(0,
                                (image.shape[0] - z_patch * z_steps) / (z_patch * (z_steps - 1))) if z_steps > 1 else 0

            # Split the image into patches
            image_indices, padding = split_image(
                image, z_patch, w_patch, h_patch, z_overlap, w_overlap, h_overlap
            )
            print("z_patch:", z_patch, "  z_overlap:", z_overlap)
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

    def save_patches_to_h5(self):
        """
        Saves the extracted patches to an HDF5 file.

        Input:
        - None

        Output:
        - None
        """
        with h5py.File(self.h5_file_path, 'w') as h5f:
            for idx, (image_idx, z_idx, w_idx, h_idx) in tqdm(
                enumerate(self.indices), total=len(self.indices), dynamic_ncols=False, desc="Saving patches to h5"
            ):
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
                h5f.create_dataset(f'patch_{idx}', data=patch.numpy())
        print(f"The temp h5f file has been saved to {self.h5_file_path} successfully!")

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
        Retrieves a single patch and its indices from the HDF5 file.

        Input:
        - item (int): Index of the patch to retrieve.

        Output:
        - tuple: Patch tensor and its corresponding indices.
        """
        patch = torch.tensor(self.h5_file[f'patch_{item}'][:])
        image_idx, z_idx, w_idx, h_idx = self.indices[item]
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
