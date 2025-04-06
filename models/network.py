# %% ##################################### 
 # Descripttion: 
 # version: 
 # Author: Yuanjie Gu @ Fudan
 # Date: 2024-10-23
 # LastEditors: Yuanjie Gu
 # LastEditTime: 2024-12-25
# %% #####################################
import torch.nn as nn
import torch.nn.functional as F
from .model_CRN import CRN as CNR
    
class HessianConstraintLoss3D(nn.Module):
    def __init__(self):
        super(HessianConstraintLoss3D, self).__init__()

    def forward(self, input_tensor):
        """
        Compute the 3D Hessian matrix constraint loss for the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input tensor with shape (B, C, Z, X, Y).
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure the input tensor has the correct shape
        assert input_tensor.dim() == 5, "Input tensor must have shape (B, C, Z, X, Y)"
        
        # Pad the input tensor to handle boundary conditions
        input_tensor_padded = F.pad(input_tensor, (1, 1, 1, 1, 1, 1), mode='replicate')
        
        # Compute the first-order partial derivatives
        dz = input_tensor_padded[:, :, 1:-1, 1:-1, 2:] - input_tensor_padded[:, :, 1:-1, 1:-1, :-2]
        dx = input_tensor_padded[:, :, 1:-1, 2:, 1:-1] - input_tensor_padded[:, :, 1:-1, :-2, 1:-1]
        dy = input_tensor_padded[:, :, 2:, 1:-1, 1:-1] - input_tensor_padded[:, :, :-2, 1:-1, 1:-1]
        
        # Compute the second-order partial derivatives
        dzz = dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :]
        dxx = dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :]
        dyy = dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1]
        
        dzx = dz[:, :, :, 1:, :-1] - dz[:, :, :, :-1, 1:]
        dzy = dz[:, :, 1:, :, :-1] - dz[:, :, :-1, :, 1:]
        dxy = dx[:, :, 1:, :, :-1] - dx[:, :, :-1, :, 1:]
        
        # Ensure all second-order derivatives have the same shape
        min_shape_z = min(dzz.shape[2], dxx.shape[2], dyy.shape[2], dzx.shape[2], dzy.shape[2], dxy.shape[2])
        min_shape_x = min(dzz.shape[3], dxx.shape[3], dyy.shape[3], dzx.shape[3], dzy.shape[3], dxy.shape[3])
        min_shape_y = min(dzz.shape[4], dxx.shape[4], dyy.shape[4], dzx.shape[4], dzy.shape[4], dxy.shape[4])
        
        dzz = dzz[:, :, :min_shape_z, :min_shape_x, :min_shape_y]
        dxx = dxx[:, :, :min_shape_z, :min_shape_x, :min_shape_y]
        dyy = dyy[:, :, :min_shape_z, :min_shape_x, :min_shape_y]
        dzx = dzx[:, :, :min_shape_z, :min_shape_x, :min_shape_y]
        dzy = dzy[:, :, :min_shape_z, :min_shape_x, :min_shape_y]
        dxy = dxy[:, :, :min_shape_z, :min_shape_x, :min_shape_y]
        
        # Compute the Frobenius norm of the Hessian matrix
        hessian_norm = (dzz**2 + dxx**2 + dyy**2 + 2*dzx**2 + 2*dzy**2 + 2*dxy**2).mean()
        
        return hessian_norm


class Network_CNR(nn.Module): # Cross-scale Recursive Network (CRN)
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        f_maps=16,
        n_groups=4,
    ):
        super(Network_CNR, self).__init__()
        self.net = CNR(in_channels=in_channels,out_channels=out_channels,n_features=f_maps,n_groups=n_groups)
    def forward(self, x):
        x = self.net(x)
        return x