import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class wConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv1d, self).__init__()       
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.kernel_size = _single(kernel_size)
        self.groups = groups
        self.dilation = _single(dilation)      
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # den trainable --> Change this initialisation for different kernel size
        self.den = nn.Parameter(torch.tensor([0.27, 0.17, 0.54]))

    def forward(self, x):
        device = self.den.device
        center = torch.ones(1, device=device, dtype=self.den.dtype)
        alfa = torch.cat([self.den, center, torch.flip(self.den, dims=[0])])
        Phi = alfa
        
        if Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {Phi.shape} must match kernel size {self.kernel_size}")
        
        weight_Phi = self.weight * Phi
        return F.conv1d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)


class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv2d, self).__init__()       
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)      
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # den trainable --> Change this initialisation for different kernel size
        self.den = nn.Parameter(torch.tensor([0.27, 0.17, 0.54]))

    def forward(self, x):
        device = self.den.device
        center = torch.ones(1, device=device, dtype=self.den.dtype)
        alfa = torch.cat([self.den, center, torch.flip(self.den, dims=[0])])
        Phi = torch.outer(alfa, alfa)
        
        if Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {Phi.shape} must match kernel size {self.kernel_size}")
        
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)


class wConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv3d, self).__init__()       
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.kernel_size = _triple(kernel_size)
        self.groups = groups
        self.dilation = _triple(dilation)          
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # den trainable --> Change this initialisation for different kernel size
        self.den = nn.Parameter(torch.tensor([0.27, 0.17, 0.54]))

    def forward(self, x):
        device = self.den.device
        center = torch.ones(1, device=device, dtype=self.den.dtype)
        alfa = torch.cat([self.den, center, torch.flip(self.den, dims=[0])])
        Phi = torch.einsum('i,j,k->ijk', alfa, alfa, alfa)
        
        if Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {Phi.shape} must match kernel size {self.kernel_size}")
        
        weight_Phi = self.weight * Phi
        return F.conv3d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
