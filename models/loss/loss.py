"""
This code is based on Open-MMLab's one.
https://github.com/open-mmlab/mmediting
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from models.loss.SSIM import SSIM
import torch.nn.functional as F


def charbonnier_loss(pred, target, weight=None, reduction='mean', sample_wise=False, eps=1e-12):
    """
    Computes the Charbonnier loss.

    Input:
    - pred (Tensor): Predicted tensor of shape (N, C, H, W).
    - target (Tensor): Ground truth tensor of shape (N, C, H, W).
    - weight (Tensor, optional): Element-wise weights. Default: None.
    - reduction (str): Specifies the reduction to apply to the output. Default: 'mean'.
    - sample_wise (bool): Whether to calculate the loss sample-wise. Default: False.
    - eps (float): A small value to avoid division by zero. Default: 1e-12.

    Output:
    - Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target) ** 2 + eps).mean()


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (a differentiable variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution".
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, eps=1e-12):
        """
        Initializes the Charbonnier loss module.

        Input:
        - loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        - reduction (str): Specifies the reduction to apply to the output. Default: 'mean'.
        - sample_wise (bool): Whether to calculate the loss sample-wise. Default: False.
        - eps (float): A small value to avoid division by zero. Default: 1e-12.

        Output:
        - None
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: ["none", "mean", "sum"]')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Forward function for Charbonnier loss.

        Input:
        - pred (Tensor): Predicted tensor of shape (N, C, H, W).
        - target (Tensor): Ground truth tensor of shape (N, C, H, W).
        - weight (Tensor, optional): Element-wise weights. Default: None.

        Output:
        - Tensor: Calculated Charbonnier loss.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction, sample_wise=self.sample_wise
        )


class WaveletLoss(nn.Module):
    """
    Custom wavelet transform loss using discrete wavelet transform (DWT).
    """

    def __init__(self, wavelet='haar'):
        """
        Initializes the WaveletLoss module.

        Input:
        - wavelet (str): Type of wavelet to use. Default: 'haar'.

        Output:
        - None
        """
        super(WaveletLoss, self).__init__()
        self.wavelet = wavelet
        self.dwt = DWTForward(J=3, mode='zero', wave=self.wavelet).cuda()

    def forward(self, predicted_frames, target_frames):
        """
        Computes the wavelet loss between predicted and target frames.

        Input:
        - predicted_frames (Tensor): Predicted frames tensor.
        - target_frames (Tensor): Target frames tensor.

        Output:
        - Tensor: Calculated wavelet loss.
        """
        loss = 0.0
        for i in range(predicted_frames.size(1)):
            predicted_wavelet = self.wavelet_transform(predicted_frames[:, i, :, :])
            target_wavelet = self.wavelet_transform(target_frames[:, i, :, :])
            loss += torch.mean((predicted_wavelet - target_wavelet) ** 2)
        return loss

    def wavelet_transform(self, image):
        """
        Performs wavelet transform on the input image.

        Input:
        - image (Tensor): Input image tensor.

        Output:
        - Tensor: Wavelet-transformed tensor.
        """
        image = torch.unsqueeze(image, dim=0)
        Yl, Yh = self.dwt(image)
        return Yl


class SSIMLoss(nn.Module):
    """
    Custom SSIM (Structural Similarity Index) loss.
    """

    def __init__(self):
        """
        Initializes the SSIMLoss module.

        Input:
        - None

        Output:
        - None
        """
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(window_size=11, size_average=True)

    def forward(self, predicted_frames, target_frames):
        """
        Computes the SSIM loss between predicted and target frames.

        Input:
        - predicted_frames (Tensor): Predicted frames tensor.
        - target_frames (Tensor): Target frames tensor.

        Output:
        - Tensor: Calculated SSIM loss.
        """
        return 1 - self.ssim(predicted_frames, target_frames)


class Projection_Loss(nn.Module):
    """
    Custom projection loss for comparing cropped regions of predicted and target tensors.
    """

    def __init__(self):
        """
        Initializes the Projection_Loss module.

        Input:
        - None

        Output:
        - None
        """
        super(Projection_Loss, self).__init__()

    def forward(self, predicted_tensor, target_tensor, crop_size=20):
        """
        Computes the projection loss between predicted and target tensors.

        Input:
        - predicted_tensor (Tensor): Predicted tensor of shape (N, T, W, H).
        - target_tensor (Tensor): Target tensor of shape (N, T, W, H).
        - crop_size (int): Size of the cropped region. Default: 20.

        Output:
        - Tensor: Calculated projection loss.
        """
        _, t, w, h = predicted_tensor.size()
        if crop_size > w or crop_size > h:
            crop_size = min(w, h)
        total_loss = 0
        for i in range(10):
            top = torch.randint(0, w - crop_size + 1, (1,))
            left = torch.randint(0, h - crop_size + 1, (1,))
            cropped_predicted_tensor = predicted_tensor[:, :, top:(top + crop_size), left:(left + crop_size)]
            cropped_target_tensor = target_tensor[:, :, top:(top + crop_size), left:(left + crop_size)]
            mean_predicted_w = cropped_predicted_tensor.mean(dim=(2, 3))
            mean_target_w = cropped_target_tensor.mean(dim=(2, 3))
            loss_w = F.l1_loss(mean_predicted_w, mean_target_w)
            total_loss += loss_w
        return total_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Custom temporal consistency loss for ensuring smooth transitions between frames.
    """

    def __init__(self):
        """
        Initializes the TemporalConsistencyLoss module.

        Input:
        - None

        Output:
        - None
        """
        super(TemporalConsistencyLoss, self).__init__()

    def forward(self, predicted_frames):
        """
        Computes the temporal consistency loss for predicted frames.

        Input:
        - predicted_frames (Tensor): Predicted frames tensor.

        Output:
        - Tensor: Calculated temporal consistency loss.
        """
        mean_diff = torch.abs(predicted_frames[:, 1:] - predicted_frames[:, :-1]).mean(dim=(2, 3))
        return mean_diff.mean()


if __name__ == '__main__':
    """
    Example usage of the Projection_Loss module.
    """
    criterion = Projection_Loss()

    # Simulated predicted frames
    predicted_frames = torch.randn(1, 200, 512, 512).cuda()

    # Simulated target frames
    target_frames = torch.randn(1, 200, 512, 512).cuda()

    # Compute loss
    loss = criterion(predicted_frames, target_frames)

    # Backpropagation
    loss.backward()
