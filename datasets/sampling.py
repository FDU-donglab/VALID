import pywt
import torch
import math
import random
import numpy as np
from einops import rearrange
from torch.nn import Module
from torch.autograd import Function
from torch.autograd import Variable
operation_seed_counter = 0

class DWTFunction_3D(Function):
    """
    Custom 3D Discrete Wavelet Transform (DWT) function for forward and backward passes.
    """

    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        """
        Forward pass for 3D DWT.

        Input:
        - ctx: Context object for saving variables for backward computation.
        - input (Tensor): Input tensor of shape (N, C, D, H, W).
        - matrix_Low_0, matrix_Low_1, matrix_Low_2 (Tensor): Low-pass filter matrices.
        - matrix_High_0, matrix_High_1, matrix_High_2 (Tensor): High-pass filter matrices.

        Output:
        - Tuple of tensors: Eight components (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH).
        """
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0 = 2, dim1 = 3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0 = 2, dim1 = 3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0 = 2, dim1 = 3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0 = 2, dim1 = 3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0 = 2, dim1 = 3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0 = 2, dim1 = 3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0 = 2, dim1 = 3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0 = 2, dim1 = 3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                      grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        """
        Backward pass for 3D DWT.

        Input:
        - ctx: Context object containing saved variables.
        - grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH (Tensor): Gradients of outputs.

        Output:
        - Tensor: Gradient of the input tensor.
        """
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HLL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HLH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HHL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HHH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None

class DWT_3D(Module):
    """
    3D Discrete Wavelet Transform (DWT) for decomposing 3D data into frequency components.
    """

    def __init__(self, wavename):
        """
        Initializes the 3D DWT module.

        Input:
        - wavename (str): Name of the wavelet to use (e.g., 'haar', 'biorx.y').

        Output:
        - None
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        Generates transformation matrices for low-pass and high-pass filters.

        Input:
        - None

        Output:
        - None
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        """
        Performs the forward pass of 3D DWT.

        Input:
        - input (Tensor): Input tensor of shape (N, C, D, H, W).

        Output:
        - Tuple of tensors: Eight components (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH).
        """
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)

class RandomSampler_0123:
    """
    Random sampler for selecting unique numbers from the set {0, 1, 2, 3}.
    """

    def __init__(self):
        """
        Initializes the random sampler.

        Input:
        - None

        Output:
        - None
        """
        self.numbers = [0, 1, 2, 3]
        random.shuffle(self.numbers)

    def draw(self):
        """
        Draws a random number from the set.

        Input:
        - None

        Output:
        - int: A randomly selected number.
        """
        if not self.numbers:
            raise AssertionError("No more numbers to draw")
        return self.numbers.pop()

def generate_mask_pair(img):
    """
    Generates random mask pairs for subimage extraction.

    Input:
    - img (Tensor): Input tensor of shape (N, T, H, W).

    Output:
    - Tuple of tensors: Four masks for subimage extraction.
    """
    n, t, h, w = img.shape
    mask1 = torch.zeros(
        size=(n * t // 2 * h // 2 * w // 2 * 8,), dtype=torch.bool, device=img.device
    )
    mask2 = torch.zeros(
        size=(n * t // 2 * h // 2 * w // 2 * 8,), dtype=torch.bool, device=img.device
    )
    mask3 = torch.zeros(
        size=(n * t // 2 * h // 2 * w // 2 * 8,), dtype=torch.bool, device=img.device
    )
    mask4 = torch.zeros(
        size=(n * t // 2 * h // 2 * w // 2 * 8,), dtype=torch.bool, device=img.device
    )
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [ 
            [0, 1, 2, 4],
            [0, 1, 3, 5],
            [1, 2, 3, 7],
            [0, 2, 3, 6],
            [0, 4, 5, 6],
            [1, 4, 5, 7],
            [2, 4, 6, 7],
            [3, 5, 6, 7],
        ],
        dtype=torch.int64,
        device=img.device
    )
    rd_idx = torch.zeros(
        size=(n * t // 2 * h // 2 * w // 2,), dtype=torch.int64, device=img.device
    )
    torch.randint(
        low=0,
        high=8,
        size=(n * t // 2 * h // 2 * w // 2,),
        generator=get_generator(),
        out=rd_idx
    )
    
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(
        start=0,
        end=n * t  // 2 * h // 2 * w // 2 * 8,
        step=8,
        dtype=torch.int64,
        device=img.device,
    ).reshape(-1, 1)
    # get masks
    rd_0123 = RandomSampler_0123()
    mask1[rd_pair_idx[:, rd_0123.draw()]] = 1
    mask2[rd_pair_idx[:, rd_0123.draw()]] = 1
    mask3[rd_pair_idx[:, rd_0123.draw()]] = 1
    mask4[rd_pair_idx[:, rd_0123.draw()]] = 1

    return mask1, mask2, mask3, mask4


def generate_subimages(img, mask):
    """
    Generates subimages from the input image using the provided mask.

    Input:
    - img (Tensor): Input tensor of shape (N, T, H, W).
    - mask (Tensor): Mask tensor for subimage extraction.

    Output:
    - Tensor: Subimages of shape (N, T/2, H/2, W/2).
    """
    n, t, h, w = img.shape
    subimage = torch.zeros(
        n , t // 2 , h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device
    )
    img_per_channel = space_to_depth(img, block_size=2)
    img_per_channel = img_per_channel.permute(0, 1, 2, 3, 4).reshape(-1)
    subimage = img_per_channel[mask].reshape(n, t // 2, h // 2, w // 2)
    return subimage


def get_generator():
    """
    Creates a random number generator with a unique seed.

    Input:
    - None

    Output:
    - torch.Generator: A random number generator.
    """
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    """
    Converts spatial dimensions into depth channels.

    Input:
    - x (Tensor): Input tensor of shape (N, T, H, W).
    - block_size (int): Block size for spatial-to-depth conversion.

    Output:
    - Tensor: Transformed tensor with depth channels.
    """
    n, t, h, w = x.shape
    x = x.view(n, t // block_size, block_size, h // block_size, block_size, w // block_size, block_size)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    x = x.view(n, t // block_size, h // block_size, w // block_size, block_size * block_size * block_size)
    return x


if __name__ == "__main__":
    """
    Example usage of the sampling functions.
    """
    img = torch.randn((2, 10, 16, 64, 64))
    mask1, mask2 = generate_mask_pair(img)

    noisy_sub1 = generate_subimages(img, mask1)
    noisy_sub2 = generate_subimages(img, mask2)
    print(noisy_sub1.shape)
    print(noisy_sub2.shape)
