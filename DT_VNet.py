from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torchsummary import summary

import pdb

from monai.utils import ensure_tuple_rep,look_up_option,optional_import
from torch.nn import LayerNorm
from collections.abc import Sequence

from monai.networks.blocks import PatchEmbed
from monai.networks.layers import trunc_normal_
# from monai.networks.blocks import MLPBlock as Mlp

from torchsummary import summary

#df
from functools import reduce
from operator import mul
from timm.models.layers import to_3tuple
from torch.nn.modules import module
NEG_INF = 1
rearrange, _ = optional_import("einops", name="rearrange")


def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(2.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

def window_partition(x, window_size):
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,)
        windows = (x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c))
    return windows

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  
        x1 = x[:, 0::2, 0::2, 1::2, :] 
        x2 = x[:, 0::2, 1::2, 0::2, :]  
        x3 = x[:, 0::2, 1::2, 1::2, :]  
        x4 = x[:, 1::2, 0::2, 0::2, :] 
        x5 = x[:, 1::2, 0::2, 1::2, :] 
        x6 = x[:, 1::2, 1::2, 0::2, :] 
        x7 = x[:, 1::2, 1::2, 1::2, :]  
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  
        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class Attention(nn.Module):
    def __init__(self, dim, group_size1, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.group_size1 = group_size1  # Gd, Gh, Gw

    def forward(self, x, group_size,mask=None,):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
#         相对位置偏差
        mesh_args = torch.meshgrid.__kwdefaults__  
        if len(group_size) == 3:
            relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * group_size[0] - 1) * (2 * group_size[1] - 1) * (2 * group_size[2] - 1),
                    self.num_heads,))
            coords_d = torch.arange(group_size[0])
            coords_h = torch.arange(group_size[1])
            coords_w = torch.arange(group_size[2])
            
            if mesh_args is not None: 
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += group_size[0] - 1
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 2] += group_size[2] - 1
            relative_coords[:, :, 0] *= (2 * group_size[1] - 1) * (2 * group_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * group_size[2] - 1
        relative_position_index = relative_coords.sum(-1) 
        trunc_normal_(relative_position_bias_table, std=0.02)
        
        relative_position_bias = relative_position_bias_table[relative_position_index.clone()
                                                                   [:N, :N].reshape(-1)].reshape(N, N, -1)            
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).cuda()
        
        if mask is not None:
            nG = mask.shape[0]
            a=attn
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x        

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)
    
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
            super()._check_input_dim(input) 
    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
    
class DCN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(DCN, self).__init__()
        self.proj = nn.Sequential(nn.Conv3d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        feat_token = x
        cnn_feat = rearrange(feat_token, 'b d h w c -> b c d h w')
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = rearrange(x, 'b c d h w -> b d h w c')
        return x 
    
class GCP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv3d(in_features, 4 * in_features, kernel_size=1, groups=in_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(4 * in_features, in_features, kernel_size=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0,4,1,2,3)  #[4,36,16, 48, 48]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.permute(0,2,3,4,1)
        x = self.drop(x)  #[4, 36, 16, 48, 48]
        return x
    
class _Skip(nn.Module):
    def __init__(self, nchan, elu,depth):
        super(_Skip, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, stride=1,padding=1)
        self.bn1 = ContBatchNorm3d(nchan)
        self.dp1=nn.Dropout3d()
        self.depth=depth

    def forward(self, x):
        if self.depth==0:
            out = self.relu1(self.bn1(self.dp1(self.conv1(x))))
        else:
            out = self.relu1(self.bn1(self.conv1(x)))
        return out
    
def _ConvSkip(nchan, depth, elu):
    layers = []
    for i in range(depth):
        layers.append(_Skip(nchan, elu,i))
        if i==1:
            layers.append(SELayer(nchan))
    return nn.Sequential(*layers)

def _Conv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(Conv(nchan, elu))
    return nn.Sequential(*layers)

class Conv(nn.Module):
    def __init__(self, nchan, elu):
        super(Conv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, stride=1,padding=1)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out
    
class SkipCon(nn.Module):
    def __init__(self,Chans, nConvs, elu, dropout=False):
        super(SkipCon, self).__init__()

        self.relu1 = ELUCons(elu, Chans)
        self.ops = _ConvSkip(Chans, nConvs, elu)

    def forward(self, x):
        out = self.ops(x)
        out = self.relu1(torch.add(out, x))
        return out
    
class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops_1 = _Conv(outChans, nConvs, elu)
        self.ops_2 = _Conv(inChans, nConvs, elu)
        
        
        self.up_conv2 = nn.ConvTranspose3d(inChans, outChans, kernel_size=1, stride=1)
        self.bn2 = ContBatchNorm3d(outChans)
        self.relu2 = ELUCons(elu, outChans)
        
        self.reduceChn=PFA(inChans,outChans,elu)
        
    
    def forward(self, x, skipx):
        out = self.do1(x)

        if (x.shape[1] != skipx.shape[1]):
            out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipx), 1)

        if xcat.shape[1]!=36:
            out=self.reduceChn(xcat)
            out = self.ops_1(out)
        else:
            out=self.ops_2(xcat)
        return out
    
class PFA(nn.Module):
    expansion = 4
    def __init__(self, inchannel,outchannel,elu):
        super(PFA, self).__init__()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv3d(inchannel,inchannel,kernel_size=3,stride=1,padding=1,groups=1,bias=False)
        self.bn_1=nn.BatchNorm3d(inchannel)

        
        self.conv3d_2 = nn.Conv3d(inchannel,outchannel,kernel_size=3,stride=1,padding=1,groups=1,bias=False)
        self.bn_2=nn.BatchNorm3d(outchannel)
        
        if inchannel==576:
            self.globalAvgPool = nn.AvgPool3d((6,6,2), stride=1)    
            self.globalMaxPool = nn.MaxPool3d((6,6,2), stride=1)        
        elif inchannel==288:
            self.globalAvgPool = nn.AvgPool3d((12,12,4), stride=1)    
            self.globalMaxPool = nn.MaxPool3d((12,12,4), stride=1)
        elif inchannel==144:
            self.globalAvgPool = nn.AvgPool3d((24,24,8), stride=1)    
            self.globalMaxPool = nn.MaxPool3d((24,24,8), stride=1)
        elif inchannel==72:
            self.globalAvgPool = nn.AvgPool3d((48,48,16), stride=1)    
            self.globalMaxPool = nn.MaxPool3d((48,48,16), stride=1)
         
        self.fc1 = nn.Linear(in_features=inchannel, out_features=outchannel)
        self.fc2 = nn.Linear(in_features=outchannel, out_features=inchannel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):  
        residual = x #[1,160,56,56]
        original_out = x
        out=x
        out1 = x #[1, 128, 56, 56]
        
        # For global average pool
        out = self.globalAvgPool(out) 
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = self.relu_1(out)
        out = self.fc2(out) 
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1,1) 
        out = out * original_out 
        # For global maximum pool
        out1 = self.globalMaxPool(out1) 
        out1 = out1.view(out1.size(0), -1) 
        out1 = self.fc1(out1) 
        out1 = self.relu_1(out1)
        out1 = self.fc2(out1) 
        out1 = self.sigmoid(out1) 
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1,1) 
        out1 = out1 * original_out 
                
        out += out1

        out=self.relu_2(out)   
        out=self.conv3d_1(out)
        out=self.bn_1(out)
        out += residual
        out=self.relu_2(out)
        
        out=self.conv3d_2(out)
        out=self.bn_2(out)
        out=self.relu_3(out) 
        return out
    
class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=1,stride=1 ,padding=0)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out
    
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 interval,
                 depth,
                 num_heads,
                 group_size=(2, 8, 8),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 i_layer=None):
        super().__init__()
        self.group_size = group_size
        
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            UTrans(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                group_size=group_size,
                interval=interval,
                qk_scale=qk_scale,
                gsm=0 if (i % 2 == 0) else 1, )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        self.dcn = DCN(in_chans=dim, embed_dim=dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.dcn(x)

        for i, blk in enumerate(self.blocks):
            if i==self.depth-1:
                bl=True
            else:
                bl=False
            x = blk(x,bl) 
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x
    
class UTrans(nn.Module):
    def __init__(self, dim, num_heads, group_size=(2, 7, 7), interval=8, gsm=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.group_size = group_size
        self.gsm = gsm
        self.interval = interval
        self.attn = Attention(
            dim,
            group_size1=self.group_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.gcp=GCP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward_part1(self, x):
        x_sw=x
        B, D, H, W, C = x.shape
        x = self.norm1(x)  #[4, 16, 48, 48, 36]
        
        if H < self.group_size[1]:  #(2, 8, 8)
            self.gsm = 0
            self.group_size = (D, H, W)
            
        size_div = self.interval if self.gsm == 1 else self.group_size
        if isinstance(size_div, int): size_div = to_3tuple(size_div)
        
        pad_l = pad_t = pad_d0 = 0
        pad_d = (size_div[0] - D % size_div[0]) % size_div[0]
        pad_b = (size_div[1] - H % size_div[1]) % size_div[1]
        pad_r = (size_div[2] - W % size_div[2]) % size_div[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape
        
        mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
        
        if pad_d > 0:
            mask[:, -pad_d:, :, :, :] = -1
        if pad_b > 0:
            mask[:, :, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, :, -pad_r:, :] = -1
        
        if self.gsm == 0: 
            B, D2, H2, W2, C = x.shape
            Gd = size_div[0]
            Gh = size_div[1]
            Gw = size_div[2]
            
            x = x.view(B, D2 // Gd, Gd, H2 // Gh, Gh, W2 // Gw, Gw, C).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            x = x.reshape(-1, reduce(mul, size_div), C)
            
            nG = (Dp * Hp * Wp) // (Gd * Gh * Gw)  # group_num

            if pad_r > 0 or pad_b > 0 or pad_d > 0:
                mask = mask.reshape(1, Dp // Gd, Gd, Hp // Gh, Gh, Wp // Gw, Gw, 1).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
                mask = mask.reshape(nG, 1, Gd * Gh * Gw)
                attn_mask = torch.zeros((nG, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
            
        else: 
            B, D2, H2, W2, C = x.shape
            Id = D2 // self.group_size[0]
            Ih = H2 // self.group_size[1]
            Iw = W2 // self.group_size[2]
            
            Gd, Gh, Gw = D2 // Id, H2 // Ih, W2 // Iw
            
            #移动窗口
            x = torch.roll(x, shifts=(-size_div[0], -size_div[1], -size_div[2]), dims=(1, 2, 3))
            B_sw, D2_sw, H2_sw, W2_sw, C_sw = x_sw.shape
            dp = int(np.ceil(D2_sw / Gd)) *  Gd
            hp = int(np.ceil(H2_sw / Gh)) *  Gh
            wp = int(np.ceil(W2_sw / Gw)) *  Gw
            mask_matrix = compute_mask([dp, hp, wp], (Gd, Gh, Gw), size_div, x.device)
            attn_mask_sw = mask_matrix
            
            x = x.reshape(B, D2 // Id, Id, H2 // Ih, Ih, W2 // Iw, Iw, C).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.reshape(B * Id * Ih * Iw, Gd * Gh * Gw, C)
            
            nG = Id * Ih * Iw  # group_num

            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gd, Id, Gh, Ih, Gw, Iw, 1).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
                mask = mask.reshape(nG, 1, Gd * Gh * Gw)
                attn_mask = torch.zeros((nG, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
            if attn_mask!=None:
                print('!!!')
            else:
                attn_mask=attn_mask_sw
            
        # multi-head self-attention
        x = self.attn(x, group_size=(Gd, Gh, Gw),mask=attn_mask) #[72, 128, 96]

        if self.gsm == 0:
            x = x.reshape(B, D2 // size_div[0], H2 // size_div[1], W2 // size_div[2], size_div[0], size_div[1],
                       size_div[2], C).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous() 
            x = x.view(B, D2, H2, W2, -1)
        else:
            x = x.reshape(B, Id, Ih, Iw,D2 // Id, H2 // Ih, W2 // Iw, C).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() 
            x = x.view(B, D2, H2, W2, -1)
            x = torch.roll(x, shifts=(size_div[0], size_div[1], size_div[2]), dims=(1, 2, 3))

        # remove padding
        if pad_d > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, bl):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        
        if bl==True:
            x=self.norm3(x)
            x=self.gcp(x)    
        return x
    
class DTdec(nn.Module):
    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 interval_list=[4, 4, 2, 1],
                 patch_size=(2, 2, 2),
                 in_chans=1,
                 embed_dim=36,
                 depths=[2, 2, 2, 2],   
                 num_heads=[3, 6, 12, 24],
                 group_size=(2, 8, 8),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.group_size = group_size
        self.patch_size = patch_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self._drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  

        # build layers  
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),  
                depth=depths[i_layer], #2 2 6 2
                num_heads=num_heads[i_layer], #3,6,12,24
                mlp_ratio=mlp_ratio, #4
                qkv_bias=qkv_bias, #true
                drop=drop_rate, #0
                attn_drop=attn_drop_rate, #0
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers else None,
                use_checkpoint=use_checkpoint,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                group_size=group_size, #2,7,7
                interval=interval_list[i_layer], #8,4,2,1
                qk_scale=qk_scale, #none
                i_layer=i_layer
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self._freeze_stages()
        self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        pos_keys = [k for k in state_dict.keys() if "pos" in k]
        for k in pos_keys:
            del state_dict[k]

        biases_keys = [k for k in state_dict.keys() if "biases" in k]
        for k in biases_keys:
            del state_dict[k]
        attn_index_keys = [k for k in state_dict.keys() if "attn" in k]
        weight = collections.OrderedDict()
        for k in attn_index_keys:
            weight[k] = state_dict[k]

        mlp_keys = [k for k in state_dict.keys() if "mlp" in k]
        for k in mlp_keys:
            weight[k] = state_dict[k]

        up_attn_keys = [k for k in self.state_dict().keys() if ("attn" or "layers_up.") in k]
        up_attn_keys = up_attn_keys[len(attn_index_keys):]
        match_attn_keys = attn_index_keys[:len(up_attn_keys)]
        s1_up = up_attn_keys[0:24]
        s1_match = match_attn_keys[16:]
        s2_up = up_attn_keys[24:32]
        s2_match = match_attn_keys[8:16]
        s3_up = up_attn_keys[32:]
        s3_match = match_attn_keys[:8]

        for i in range(len(s1_up)):
            weight[s1_up[i]] = state_dict[s1_match[i]]
        for i in range(len(s2_up)):
            weight[s2_up[i]] = state_dict[s2_match[i]]
        for i in range(len(s3_up)):
            weight[s3_up[i]] = state_dict[s3_match[i]]
        self.load_state_dict(weight, strict=False)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if self.pretrained != None:
            if isinstance(self.pretrained, str):
                self.apply(_init_weights)
                if self.pretrained2d:
                    self.inflate_weights()

    def forward_features(self, x): 
        x_downsample = []        
        x = self.patch_embed(x) #[1, 36, 16, 48, 48]
        x = self._drop(x) #[1, 36, 16, 48, 48]
        x_downsample.append(x)
        for layer in self.layers:
            x = layer(x.contiguous())  
            x_downsample.append(x)
        return x_downsample  
    
    def forward(self, x):
        x=x.permute(0,1,4,3,2)
        x_downsample = self.forward_features(x) 
        return x_downsample
    
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 2, 2), in_chans=1, embed_dim=36, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x
    
class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1,outChans,kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x18 = torch.cat((x, x, x, x, x, x, x, x,x, x, x, x, x, x, x, x,x, x), 1) 
        out = self.relu1(torch.add(out, x18))
        return out
    
class DTVNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        spatial_dims: int = 3,
        feature_size: int = 24,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        downsample="merging",
        use_v2=False,
        normalize: bool = True,
        elu=True, 
        nll=False
    )-> None:
        super().__init__()
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        
        self.normalize = normalize
        
        self.in_tr = InputTransition(18, elu)
        self.Skip2=SkipCon(36,1, elu, dropout=True)
        self.Skip3=SkipCon(72,2, elu, dropout=True)
        self.Skip4=SkipCon(144,2, elu, dropout=True)
        self.Skip5=SkipCon(288,3, elu, dropout=True)
        self.Skip6=SkipCon(576,2, elu, dropout=True)
        
        self.up5=UpTransition(576, 288, 2, elu, dropout=True)
        self.up4=UpTransition(288, 144, 2, elu, dropout=True)
        self.up3=UpTransition(144, 72, 2, elu, dropout=True)
        self.up2=UpTransition(72, 36, 1, elu)
        self.up1=UpTransition(36, 18, 1, elu)
        self.out=OutputTransition(36, elu, nll)
        
        self.df = DTdec()

    def forward(self, x):  
        hidden_states_out = self.df(x)
        e1 = self.in_tr(x) #[4, 18, 96, 96, 32]
        
        e2=hidden_states_out[0]
        e2 = self.Skip2(e2.permute(0,1,4,3,2)) #[4, 36, 48, 48, 16]
        
        e3=hidden_states_out[1]
        e3 = self.Skip3(e3.permute(0,1,4,3,2)) #[4, 72, 24, 24, 8]
        
        e4=hidden_states_out[2]
        e4 = self.Skip4(e4.permute(0,1,4,3,2)) #[4, 144, 12, 12, 4]
        
        e5=hidden_states_out[3]
        e5 = self.Skip5(e5.permute(0,1,4,3,2)) #[4, 288, 6, 6, 2]
        
        e6=hidden_states_out[4].permute(0,1,4,3,2)
        
        d5=self.up5(e6,e5) #[4, 288, 6, 6, 2]
        d4=self.up4(d5,e4) #[4, 144, 12, 12, 4]
        d3=self.up3(d4,e3) #[4, 72, 24, 24, 8]
        d2=self.up2(d3,e2) #[4, 36, 48, 48, 16]
        d1=self.up1(d2,e1) #[4, 36, 96, 96, 32]
        out=self.out(d1) #[4, 2, 96, 96, 32]
        return out
    
