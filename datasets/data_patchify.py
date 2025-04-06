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
    :return: Index list, shape (num_patches, 4), each element is a quadruplet (image_idx, t_idx, w_idx, h_idx)
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
    padded_image = torch.nn.functional.pad(
        image, (0, h_pad, 0, w_pad, 0, t_pad), mode="constant", value=0
    )

    # Calculate indices
    indices = []
    for t_idx in range(0, padded_image.shape[0] - t_patch + 1, t_stride):
        for w_idx in range(0, padded_image.shape[1] - w_patch + 1, w_stride):
            for h_idx in range(0, padded_image.shape[2] - h_patch + 1, h_stride):
                indices.append(
                    (0, t_idx, w_idx, h_idx)
                )  # 0 is the image index, will be updated later

    return indices, (t_pad, w_pad, h_pad)


def reconstruct_image(
    patches,
    indices,
    original_shapes,
    t_patch,
    w_patch,
    h_patch,
    t_overlap,
    w_overlap,
    h_overlap,
    paddings,
):
    """
    Reconstruct the original images from the split patches.

    :param patches: List of split patches, shape (num_patches, t_patch, w_patch, h_patch)
    :param indices: List of indices of patches in the original images, shape (num_patches, 4)
    :param original_shapes: List of shapes of the original images [(t1, w1, h1), (t2, w2, h2), ...]
    :param t_patch: Patch size in the time dimension
    :param w_patch: Patch size in the width dimension
    :param h_patch: Patch size in the height dimension
    :param t_overlap: Overlap ratio in the time dimension (0~1)
    :param w_overlap: Overlap ratio in the width dimension (0~1)
    :param h_overlap: Overlap ratio in the height dimension (0~1)
    :param paddings: List of paddings of the original images [(t_pad1, w_pad1, h_pad1), (t_pad2, w_pad2, h_pad2), ...]
    :return: List of reconstructed images, shape [(t1, w1, h1), (t2, w2, h2), ...]
    """
    reconstructed_images = []
    for image_idx, original_shape in enumerate(original_shapes):
        t, w, h = original_shape
        t_pad, w_pad, h_pad = paddings[image_idx]
        # Calculate overlap strides
        t_stride = int(t_patch * (1 - t_overlap))
        w_stride = int(w_patch * (1 - w_overlap))
        h_stride = int(h_patch * (1 - h_overlap))

        # Initialize the reconstructed image
        reconstructed_image = torch.zeros(
            (t + t_pad, w + w_pad, h + h_pad),
            dtype=patches.dtype,
            device=patches.device,
        )
        patch_count = torch.zeros_like(reconstructed_image)

        # Place patches back to their original positions
        for i, (idx, t_idx, w_idx, h_idx) in enumerate(indices):
            if idx == image_idx:
                reconstructed_image[
                    t_idx : t_idx + t_patch,
                    w_idx : w_idx + w_patch,
                    h_idx : h_idx + h_patch,
                ] += patches[i]
                patch_count[
                    t_idx : t_idx + t_patch,
                    w_idx : w_idx + w_patch,
                    h_idx : h_idx + h_patch,
                ] += 1

        # Calculate the average
        reconstructed_image /= patch_count

        # Remove padding
        reconstructed_image = reconstructed_image[:t, :w, :h]

        reconstructed_images.append(reconstructed_image)

    return reconstructed_images


def main():
    from skimage import io
    import os
    import numpy as np

    # Generate multiple random images with different shapes
    images = [
        torch.randn(50, 100, 100),
        torch.randn(60, 110, 120),
        torch.randn(70, 130, 140),
    ]

    # Set patch sizes and overlap ratios
    t_patch, w_patch, h_patch = 3, 11, 13
    t_overlap, w_overlap, h_overlap = 0.5, 0.5, 0.5

    # Split the images and get indices
    indices = []
    paddings = []
    for image_idx, image in enumerate(images):
        image_indices, padding = split_image(
            image, t_patch, w_patch, h_patch, t_overlap, w_overlap, h_overlap
        )
        indices.extend(
            [
                (image_idx, t_idx, w_idx, h_idx)
                for (_, t_idx, w_idx, h_idx) in image_indices
            ]
        )
        paddings.append(padding)

    # Get patches and ensure all patches have the same size
    patches = []
    for image_idx, image in enumerate(images):
        t_pad, w_pad, h_pad = paddings[image_idx]
        padded_image = torch.nn.functional.pad(
            image, (0, h_pad, 0, w_pad, 0, t_pad), mode="constant", value=0
        )
        for t_idx, w_idx, h_idx in [
            (t_idx, w_idx, h_idx)
            for (idx, t_idx, w_idx, h_idx) in indices
            if idx == image_idx
        ]:
            patch = padded_image[
                t_idx : t_idx + t_patch,
                w_idx : w_idx + w_patch,
                h_idx : h_idx + h_patch,
            ]
            patches.append(patch)
    patches = torch.stack(patches)

    # Reconstruct the images
    reconstructed_images = reconstruct_image(
        patches,
        indices,
        [image.shape for image in images],
        t_patch,
        w_patch,
        h_patch,
        t_overlap,
        w_overlap,
        h_overlap,
        paddings,
    )

    # Compare the original images and the reconstructed images
    i = 0
    for original_image, reconstructed_image in zip(images, reconstructed_images):
        i = i + 1
        print(
            "Are the original image and the reconstructed image the same:",
            torch.allclose(original_image, reconstructed_image),
        )
        print(
            "Difference between the original image and the reconstructed image:",
            torch.norm(original_image - reconstructed_image).item(),
        )
        base_path = "/data/root/yuanjie/OCT/Denoising/FN2N/result/"
        result_path_in = os.path.join(base_path, f"in_{i}.tif")
        result_path_out = os.path.join(base_path, f"out_{i}.tif")
        io.imsave(result_path_in, original_image.numpy())
        io.imsave(result_path_out, reconstructed_image.numpy())


if __name__ == "__main__":
    main()
