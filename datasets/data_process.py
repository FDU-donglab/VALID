from matplotlib import pyplot as plt
from skimage import io
import torch
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def space_to_depth(x, block_size):
    """
    Converts spatial dimensions into depth channels.

    Input:
    - x (Tensor): Input tensor of shape (N, C, H, W).
    - block_size (int): Block size for spatial-to-depth conversion.

    Output:
    - Tensor: Transformed tensor with depth channels.
    """
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def generate_subimages(img, masks):
    """
    Generates subimages from the input image using the provided masks.

    Input:
    - img (Tensor): Input tensor of shape (N, C, H, W).
    - masks (list): List of masks for subimage extraction.

    Output:
    - Tensor: Subimages of shape (N, C, H/2, W/2).
    """
    n, c, h, w = img.shape
    subimages = torch.zeros(
        n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device
    )

    for i in range(c):
        mask = masks[i]
        img_per_channel = space_to_depth(img[:, i : i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimages[:, i : i + 1, :, :] = (
            img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
        )

    return subimages


def neighbor2neighbor(img: torch.Tensor, operation_seed_counter):
    """
    Generates random mask pairs for neighbor-to-neighbor training.

    Input:
    - img (Tensor): Input tensor of shape (N, C, H, W).
    - operation_seed_counter (int): Seed for random number generation.

    Output:
    - Tuple of lists: Two lists of masks for subimage extraction.
    """
    n, c, h, w = img.shape
    device = img.device

    masks1 = []
    masks2 = []

    for i in range(c):
        mask1 = torch.zeros(
            size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=device
        )
        mask2 = torch.zeros(
            size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=device
        )

        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=device,
        )

        g_cpu_generator = torch.Generator()
        g_cpu_generator.manual_seed(operation_seed_counter + i)
        rd_idx = torch.randint(
            low=0, high=8, size=(n * h // 2 * w // 2,), generator=g_cpu_generator
        )

        rd_idx = rd_idx.to(device)

        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(
            start=0,
            end=n * h // 2 * w // 2 * 4,
            step=4,
            dtype=torch.int64,
            device=device,
        ).reshape(-1, 1)

        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1

        masks1.append(mask1)
        masks2.append(mask2)

    return masks1, masks2


def load3Dnpy2Tensor(dataPath: str, dataExtension: str) -> torch.Tensor:
    """
    Loads a 3D image from a file and converts it to a tensor.

    Input:
    - dataPath (str): Path to the image file.
    - dataExtension (str): File extension ('tif' or 'npy').

    Output:
    - Tensor: Loaded 3D image as a tensor.
    """
    imageAll = []
    imName = os.path.basename(dataPath)
    print("load image name -----> ", imName)
    imDir = dataPath
    if dataExtension == "tif":
        image = io.imread(imDir)
    elif dataExtension == "npy":
        image = np.load(imDir)

    print(image.shape)
    imageTensor = torch.from_numpy(image / 1.0).float()
    imageAll.append(imageTensor)

    return imageAll


def load3DImages2Tensor(dataPath: str, dataExtension: str, dataNum=-1) -> torch.Tensor:
    """
    Loads multiple 3D images from a directory and converts them to tensors.

    Input:
    - dataPath (str): Path to the directory containing image files.
    - dataExtension (str): File extension ('tif' or 'npy').
    - dataNum (int): Number of images to load. Default: -1 (load all).

    Output:
    - list: List of loaded 3D images as tensors.
    """
    imageAll = []
    target_files = [f for f in os.listdir(dataPath) if f.endswith(f".{dataExtension}")]
    for imName in target_files:
        print("load image name -----> ", imName)
        imDir = dataPath + "//" + imName
        if dataExtension == "tif":
            image = io.imread(imDir)
            image = np.clip(image, 0, 65535)
        elif dataExtension == "npy":
            image = np.load(imDir)
            image = np.clip(image, 0, 65535)
        print(image.shape)
        imageTensor = torch.from_numpy(image / 1.0).float()
        if dataNum != -1:
            imageTensor = imageTensor[:dataNum]
        imageAll.append(imageTensor)

    return imageAll


def split_into_patches(x, patchw, patchh, overlap):
    """
    Splits a tensor into overlapping patches.

    Input:
    - x (Tensor): Input tensor of shape (T, C, W, H).
    - patchw (int): Width of each patch.
    - patchh (int): Height of each patch.
    - overlap (float): Overlap ratio between patches.

    Output:
    - Tuple: List of patches and their corresponding ranges.
    """
    t, c, w, h = x.shape
    stepw = int(patchw * (1 - overlap))
    steph = int(patchh * (1 - overlap))
    patches = []
    ranges = []
    for i in range(0, w - patchw + 1, stepw):
        for j in range(0, h - patchh + 1, steph):
            patch = x[:, :, i : i + patchw, j : j + patchh]
            patch_range = {
                "t_start": 0,
                "t_end": t,
                "w_start": i,
                "w_end": i + patchw,
                "h_start": j,
                "h_end": j + patchh,
            }
            patches.append(patch)
            ranges.append(patch_range)
    return patches, ranges


def process_patch(patch):
    """
    Processes a single patch.

    Input:
    - patch (Tensor): Input patch tensor.

    Output:
    - Tensor: Processed patch tensor.
    """
    processed_patch = patch * 2
    return processed_patch


def reconstruct_data(patches, shape):
    """
    Reconstructs the original data from patches.

    Input:
    - patches (list): List of patches and their corresponding ranges.
    - shape (tuple): Shape of the original data (T, C, W, H).

    Output:
    - Tensor: Reconstructed data tensor.
    """
    t, c, w, h = shape
    result = torch.zeros(shape)

    for patch, patch_range in patches:
        result[
            :,
            :,
            patch_range["w_start"] : patch_range["w_end"],
            patch_range["h_start"] : patch_range["h_end"],
        ] = patch

    return result


def get_gaussian(s, sigma=1.0 / 8) -> np.ndarray:
    """
    Generates a Gaussian map.

    Input:
    - s (tuple): Shape of the Gaussian map.
    - sigma (float): Standard deviation of the Gaussian. Default: 1.0/8.

    Output:
    - np.ndarray: Generated Gaussian map.
    """
    temp = np.zeros(s)
    coords = [i // 2 for i in s]
    sigmas = [i * sigma for i in s]
    temp[tuple(coords)] = 1
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode="constant", cval=0)
    gaussian_map /= np.max(gaussian_map)
    return gaussian_map


if __name__ == "__main__":
    """
    Example usage of the data processing functions.
    """
    train_folder = r"E:\Denoising\Deep3D\data\train\highActivity"
    imageAll_Tensor = load3DImages2Tensor(dataPath=train_folder)
    mask1, mask2 = neighbor2neighbor(
        torch.unsqueeze(imageAll_Tensor[0], dim=0), operation_seed_counter=0
    )
    sub1 = generate_subimages(torch.unsqueeze(imageAll_Tensor[0], dim=0), mask1)
    sub2 = generate_subimages(imageAll_Tensor[0], mask2)
    x = torch.randn(10, 2, 64, 64)
    patchw, patchh = 3, 3
    overlap = 0.5

    patches, ranges = split_into_patches(x, patchw, patchh, overlap)

    processed_patches = []

    for i in range(len(patches)):
        processed = process_patch(patches[i])
        processed_patches.append((processed, ranges[i]))

    reconstructed_data = reconstruct_data(processed_patches, x.shape)

    print(
        torch.allclose(reconstructed_data, x * 2)
    )

    img = io.imread(
        r"E:\Denoising\Deep3D\result\models_neurofinder_02.00\Substack (1).tif"
    )
    img = np.array(img)
    patchSize = np.array((64, 64))
    patchStride = patchSize // 2

    result = np.zeros(img.shape)
    normalization = np.zeros(img.shape)
    gaussian_map = get_gaussian(patchSize, sigma=0.5)
    for i in range(0, img.shape[0] - patchSize[0] + 1, patchStride[0]):
        for j in range(0, img.shape[1] - patchSize[1] + 1, patchStride[1]):
            patch = img[i : i + patchSize[0], j : j + patchSize[1]].astype(np.float32)
            normalization[i : i + patchSize[0], j : j + patchSize[1]] += gaussian_map
            result[i : i + patchSize[0], j : j + patchSize[1]] += patch
    plt.imshow(result, cmap=plt.get_cmap("gray"))
    plt.xticks([])
    plt.yticks([])
    plt.show()
