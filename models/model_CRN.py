# %% ##################################### 
 # Descripttion: 
 # version: 
 # Author: Yuanjie Gu @ Fudan
 # Date: 2024-10-26
 # LastEditors: Yuanjie Gu
 # LastEditTime: 2024-12-28
# %% #####################################
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np

class SE3DBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        """
        Initializes the SE3D block.

        Input:
        - channels (int): Number of input feature map channels.
        - reduction_ratio (int): Reduction ratio for channel compression.

        Output:
        - None
        """
        super(SE3DBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the SE3D block.

        Input:
        - x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Output:
        - torch.Tensor: Recalibrated tensor of the same shape as input.
        """
        b, c, d, h, w = x.size()
        z = self.global_pool(x).view(b, c)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        s = self.sigmoid(z).view(b, c, 1, 1, 1)
        return x * s

class CRN(nn.Module):
    def __init__(self, in_channels, out_channels, n_features, n_groups):
        """
        Initializes the CRN model.

        Input:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - n_features (int): Base number of features for convolution layers.
        - n_groups (int): Number of groups for GroupNorm.

        Output:
        - None
        """
        super(CRN, self).__init__()
        
        self.conv_in = nn.Conv3d(in_channels, n_features, kernel_size=(1, 1, 1), stride=1)
        self.conv_d1 = nn.Sequential(nn.ReflectionPad3d((1)),
                                     nn.Conv3d(n_features, n_features*8, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*8),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*8,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d((1)),
                                     nn.Conv3d(n_features*8, n_features*8, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*8),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*8,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.AvgPool3d(8)
                                     )

        self.conv_d2 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features, n_features*4, kernel_size=(3, 3, 3), stride=1), 
                                     SE3DBlock(n_features*4),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*4,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*4, n_features*4, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*4),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*4,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.AvgPool3d(4)
                                     )
        
        self.conv_d3 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features, n_features*2, kernel_size=(3, 3, 3), stride=1), 
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*2, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.AvgPool3d(2)
                                     )
                 
        self.conv_u1 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*12, n_features*4, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*4),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*4,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*4, n_features*4, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*4),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*4,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))
                                    
        self.conv_u2 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*6, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*2, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))
        
        self.conv_uu1 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*8, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*2, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))
        
        self.conv_b1 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*3, n_features, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features, n_features, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))
        
        self.conv_b1 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*3, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*2, n_features, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))
        
        self.conv_b2 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*3, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*2, n_features, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))
        
        self.conv_b3 = nn.Sequential(nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*3, n_features*2, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features*2),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features*2,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True),
                                     nn.ReflectionPad3d(1),
                                     nn.Conv3d(n_features*2, n_features, kernel_size=(3, 3, 3), stride=1),
                                     SE3DBlock(n_features),
                                     nn.GroupNorm(num_groups=n_groups, num_channels=n_features,eps=1e-5,affine=False),
                                     nn.ELU(inplace=True))

        self.out = nn.Conv3d(n_features, out_channels, kernel_size=(1, 1, 1), stride=1)
        self._init_kaiming_3dconv()
        
    def forward(self, x):
        """
        Forward pass for the CRN model.

        Input:
        - x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Output:
        - torch.Tensor: Output tensor of the same shape as input.
        """
        x = self.conv_in(x)

        down1 = self.conv_d1(x)
        down2 = self.conv_d2(x)
        down3 = self.conv_d3(x)
        

        down1_features = self._upsample_to_match(down1, down2)
        down2_features = self._upsample_to_match(down2, down3)
        down3_features = self._upsample_to_match(down3, x)

        fused1 = self.conv_u1(torch.cat([down2, down1_features], dim=1))
        fused2 = self.conv_u2(torch.cat([down3, down2_features], dim=1))

        fused1_features = self._upsample_to_match(fused1, fused2)
        fused2_features = self._upsample_to_match(fused2, x)

        fused_final = self.conv_uu1(torch.cat([down3, fused1_features, fused2], dim=1))
        fused_final = self._upsample_to_match(fused_final, x)

        x = torch.cat([x, down3_features], dim=1)
        x = self.conv_b1(x)
        x = torch.cat([x, fused2_features], dim=1)
        x = self.conv_b2(x)
        x = torch.cat([x, fused_final], dim=1)
        x = self.conv_b3(x)
        x = self.out(x)
        return x

    def _upsample_to_match(self, source, target):
        """
        Upsamples the source tensor to match the spatial dimensions of the target tensor.

        Input:
        - source (torch.Tensor): Source tensor to be upsampled.
        - target (torch.Tensor): Target tensor with desired spatial dimensions.

        Output:
        - torch.Tensor: Upsampled source tensor.
        """
        if source.shape[2:] != target.shape[2:]:
            return nn.functional.interpolate(source, size=target.shape[2:], mode='trilinear', align_corners=False)
        return source
    
    def _init_kaiming_3dconv(self):
        """
        Initializes all 3D convolution layers in the network with Kaiming initialization.

        Input:
        - None

        Output:
        - None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes a double 3D convolution block.

        Input:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.

        Output:
        - None
        """
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass for the double 3D convolution block.

        Input:
        - x (torch.Tensor): Input tensor.

        Output:
        - torch.Tensor: Output tensor after two 3D convolutions.
        """
        return self.double_conv(x)
    
def count_parameters_in_m(model):
    """
    Counts the number of trainable parameters in the model.

    Input:
    - model (nn.Module): The PyTorch model.

    Output:
    - None (prints the total number of parameters in millions).
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_m = total_params / 1e6
    print(f"Total Parameters: {total_params_in_m:.4f}M")

model = CRN(in_channels=1, out_channels=1, n_features=16, n_groups=2)
count_parameters_in_m(model)