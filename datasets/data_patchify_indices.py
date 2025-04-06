import torch

def split_image(image, t_patch, w_patch, h_patch, t_overlap, w_overlap, h_overlap):
    """
    Split the image into multiple patches and return the indices of the patches in the original image.

    :param image: Input image, shape (t, w, h)
    :param t_patch: Patch size in the time dimension
    :param w_patch: Patch size in the width dimension
    :param h_patch: Patch size in the height dimension
    :param t_overlap: Overlap ratio in the time dimension (0~1)
    :param w_overlap: Overlap ratio in the width dimension (0~1)
    :param h_overlap: Overlap ratio in the height dimension (0~1)
    :return: Index list, shape (num_patches, 3), each element is a triplet (t_idx, w_idx, h_idx)
    """
    t, w, h = image.shape
    
    # Calculate overlap strides
    t_stride = int(t_patch * (1 - t_overlap))
    w_stride = int(w_patch * (1 - w_overlap))
    h_stride = int(h_patch * (1 - h_overlap))
    
    # Calculate padding
    t_pad = (t_stride - (t - t_patch) % t_stride) % t_stride
    w_pad = (w_stride - (w - w_patch) % w_stride) % w_stride
    h_pad = (h_stride - (h - h_patch) % h_stride) % h_stride
    
    # Pad the image
    padded_image = torch.nn.functional.pad(image, (0, h_pad, 0, w_pad, 0, t_pad), mode='constant', value=0)
    
    # Calculate indices
    indices = []
    for t_idx in range(0, t + t_pad - t_patch + 1, t_stride):
        for w_idx in range(0, w + w_pad - w_patch + 1, w_stride):
            for h_idx in range(0, h + h_pad - h_patch + 1, h_stride):
                indices.append((t_idx, w_idx, h_idx))
    
    return indices, (t_pad, w_pad, h_pad)

def reconstruct_image(patches, indices, original_shape, t_patch, w_patch, h_patch, t_overlap, w_overlap, h_overlap, padding):
    """
    Reconstruct the original image from the split patches.

    :param patches: List of split patches, shape (num_patches, t_patch, w_patch, h_patch)
    :param indices: List of indices of patches in the original image, shape (num_patches, 3)
    :param original_shape: Shape of the original image (t, w, h)
    :param t_patch: Patch size in the time dimension
    :param w_patch: Patch size in the width dimension
    :param h_patch: Patch size in the height dimension
    :param t_overlap: Overlap ratio in the time dimension (0~1)
    :param w_overlap: Overlap ratio in the width dimension (0~1)
    :param h_overlap: Overlap ratio in the height dimension (0~1)
    :param padding: Padding of the original image (t_pad, w_pad, h_pad)
    :return: Reconstructed image, shape (t, w, h)
    """
    t, w, h = original_shape
    t_pad, w_pad, h_pad = padding
    
    # Calculate overlap strides
    t_stride = int(t_patch * (1 - t_overlap))
    w_stride = int(w_patch * (1 - w_overlap))
    h_stride = int(h_patch * (1 - h_overlap))
    
    # Initialize the reconstructed image
    reconstructed_image = torch.zeros((t + t_pad, w + w_pad, h + h_pad), dtype=patches.dtype, device=patches.device)
    patch_count = torch.zeros_like(reconstructed_image)
    
    # Place patches back to their original positions
    for i, (t_idx, w_idx, h_idx) in enumerate(indices):
        reconstructed_image[t_idx:t_idx+t_patch, w_idx:w_idx+w_patch, h_idx:h_idx+h_patch] += patches[i]
        patch_count[t_idx:t_idx+t_patch, w_idx:w_idx+w_patch, h_idx:h_idx+h_patch] += 1
    
    # Calculate the average
    reconstructed_image /= patch_count
    
    # Remove padding
    reconstructed_image = reconstructed_image[:t, :w, :h]
    
    return reconstructed_image

def main():
    # Generate a random image
    t, w, h = 50, 100, 100
    image = torch.randn(t, w, h)
    
    # Set patch sizes and overlap ratios
    t_patch, w_patch, h_patch = 3, 11, 13
    t_overlap, w_overlap, h_overlap = 0.5, 0.5, 0.5
    
    # Split the image and get indices
    indices, padding = split_image(image, t_patch, w_patch, h_patch, t_overlap, w_overlap, h_overlap)
    
    # Get patches and ensure all patches have the same size
    patches = []
    t_pad, w_pad, h_pad = padding
    padded_image = torch.nn.functional.pad(image, (0, h_pad, 0, w_pad, 0, t_pad), mode='constant', value=0)
    for t_idx, w_idx, h_idx in indices:
        patch = padded_image[t_idx:t_idx+t_patch, w_idx:w_idx+w_patch, h_idx:h_idx+h_patch]
        patches.append(patch)
    patches = torch.stack(patches)
    
    # Reconstruct the image
    reconstructed_image = reconstruct_image(patches, indices, image.shape, t_patch, w_patch, h_patch, t_overlap, w_overlap, h_overlap, padding)
    
    # Compare the original image and the reconstructed image
    print("Are the original image and the reconstructed image the same:", torch.allclose(image, reconstructed_image))
    print("Difference between the original image and the reconstructed image:", torch.norm(image - reconstructed_image).item())

if __name__ == "__main__":
    main()