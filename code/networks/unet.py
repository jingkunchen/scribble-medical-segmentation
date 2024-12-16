# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.down4_1 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//2, 0)
        self.down4_2 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//8, 0)
        self.down4_semantic_1 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//2, 0)
        self.down4_semantic_2 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//8, 0)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x4_1 = self.down4_1(x4)
        x4_2 = self.down4_2(x4)
        x4_s_1 = self.down4_semantic_1(x4)
        x4_s_2 = self.down4_semantic_2(x4)
        # return [x0, x1, x2, x3, x4]
        return [x0, x1, x2, x3, x4, x4_1, x4_2, x4_s_1, x4_s_2]


class Encoder_matching(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.down4_1 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//2, 0)
        self.down4_2 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//8, 0)
        self.down4_semantic_1 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//2, 0)
        self.down4_semantic_2 = ConvBlock(self.ft_chns[4], self.ft_chns[4]//8, 0)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x4_1 = self.down4_1(x4)
        x4_2 = self.down4_2(x4)
        x4_s_1 = self.down4_semantic_1(x4)
        x4_s_2 = self.down4_semantic_2(x4)
        return [x0, x1, x2, x3, x4, x4_1, x4_2, x4_s_1, x4_s_2]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = None
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -10000)
        print("dots:",dots.shape)
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
class SPT_superpixel(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = 64 * 64 * 64 #16 128
        # self.to_patch_tokens = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim)
        # )
        
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('i a b j h -> i a (b j h)'),
            nn.Linear(patch_dim, dim),
        )
        # 2 64 64 64 -> 2 4 4 (16 * 16 * 64)
    def forward(self, x):
        # print("x:",x.shape)
        #SPT ([2, 64, 64, 64]) -> ([2, 256, 1024])
        # shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        # shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        # x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_embedding(x)
    
class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels #4 *4* 5 * 64

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

        
        
    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        # print("x_with_shifts:",x_with_shifts.shape)
        return self.to_patch_tokens(x_with_shifts)

class Transformer_Decoder(nn.Module):
    def __init__(self, transformer_params):
        super(Transformer_Decoder, self).__init__()
        self.transformer_params = transformer_params
        self.dim = self.transformer_params['dim']
        self.depth = self.transformer_params['depth']
        self.heads = self.transformer_params['heads']
        self.dim_head = self.transformer_params['dim_head']
        self.mlp_dim = self.transformer_params['mlp_dim']
        self.num_classes = self.transformer_params['num_classes']
        self.num_patches = self.transformer_params['num_patches']
        self.pool = self.transformer_params['pool'] #'cls'
        emb_dropout = 0.1
        patch_size = 4
        
        channels = 32
        # self.num_patches = (64 // patch_size) * (64 // patch_size) 
        self.num_patches = 256
       
        #12 *64 *64 * 64
        
        self.to_patch_embedding = SPT_superpixel(dim = self.dim, patch_size = patch_size, channels = channels)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1,  self.dim))
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches,  self.dim))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1,  2*self.dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(2*self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, emb_dropout)

        self.pool = self.pool
        self.to_latent = nn.Identity()
        print("self.dim:",self.dim)
        self.superpixel_position = nn.Sequential(
            # Rearrange('b n h w -> b n (h w)'),
            nn.Linear(2*self.dim, self.dim)
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2*self.dim),
            nn.Linear( 2*self.dim, self.num_classes)
        )
        # assert (len(self.ft_chns) == 5)

    def negative_index_sampler(self, samp_num, seg_num_list):
        negative_index = []
        for i in range(samp_num.shape[0]):
            for j in range(samp_num.shape[1]):
                negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                    high=sum(seg_num_list[:j+1]),
                                                    size=int(samp_num[i, j])).tolist()
        return negative_index

    def forward(self, image, feature, superpixel):

        print("superpixel:",superpixel.shape, superpixel.dtype, torch.unique(superpixel ,return_counts = True))
        print("superpixel max:", torch.max(superpixel, dim=0))
        print("superpixel max2:", torch.max(superpixel.flatten(1,2), dim=1))

        
        superpixel_onehot = F.one_hot(superpixel.long(), num_classes=256)
        print("superpixel_onehot:",superpixel_onehot.shape)
        print("feature[2]:",feature[2].shape)
        superpixel_within_patch = torch.einsum('ijha, ibjh -> iabjh', superpixel_onehot, feature[2])
        x = self.to_patch_embedding(superpixel_within_patch) #[12, 32, 128, 128] -> [12, 32, 1024]
        localfeat = x
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 1,  self.dim).repeat(1,localfeat.shape[1], 1)
     
        x = torch.cat([x, localfeat], 2)
        print("x aftercat:",x.shape)
        b, n, _ = x.shape
        
        #class token
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        # x = torch.cat((cls_tokens, x), dim=1)

        print("superpixel:",superpixel.shape)
        print("x:",x.shape)
        superpixel_tmp = self.superpixel_position(x.to(torch.float32))
        print("superpixel_tmp:",superpixel_tmp.shape)
        # x += self.pos_embedding  
        # x += superpixel_tmp 
        x = self.dropout(x) 
        x = self.transformer(x)
        
        print("after transforer x shape:",x.shape)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        self.pool = 'mean'
        
        x = self.to_latent(x)
        print("to_latent x shape:",x.shape)
        after_mlp = self.mlp_head(x)
        print("after_mlp x shape:",after_mlp.shape)
        return after_mlp


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URDS(nn.Module):
    def __init__(self, params):
        super(Decoder_URDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        return main_seg, aux_seg1
   

class UNet_Matching_All_Queue(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Matching_All_Queue, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        dim = 128
        self.layer_number = 4
        self.K = 1000
        #self.register_buffer('sematic_embedding_0', torch.randn(1, dim))
        #self.register_buffer('sematic_embedding_1', torch.randn(1, dim))
        #self.register_buffer('sematic_embedding_2', torch.randn(1, dim))
        #self.register_buffer('sematic_embedding_3', torch.randn(1, dim))
        #dim = 32
        #self.register_buffer('sematic_embedding_0_0', torch.randn(1, dim))
        #self.register_buffer('sematic_embedding_1_0', torch.randn(1, dim))
        #self.register_buffer('sematic_embedding_2_0', torch.randn(1, dim))
        #self.register_buffer('sematic_embedding_3_0', torch.randn(1, dim))
        
        # create the semantic queue
        dim = 128
        self.register_buffer("queues0", torch.randn(dim, self.K))
        self.register_buffer("queues0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queues1", torch.randn(dim, self.K))
        self.register_buffer("queues1_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queues2", torch.randn(dim, self.K))
        self.register_buffer("queues2_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queues3", torch.randn(dim, self.K))
        self.register_buffer("queues3_ptr", torch.zeros(1, dtype=torch.long))
        dim = 32
        self.register_buffer("queues0_0", torch.randn(dim, self.K))
        self.register_buffer("queues0_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queues1_0", torch.randn(dim, self.K))
        self.register_buffer("queues1_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queues2_0", torch.randn(dim, self.K))
        self.register_buffer("queues2_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queues3_0", torch.randn(dim, self.K))
        self.register_buffer("queues3_0_ptr", torch.zeros(1, dtype=torch.long))
        

        # create the pixel queue
        dim = 128
        self.register_buffer("queue0", torch.randn(dim, self.K))
        self.register_buffer("queue0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue1", torch.randn(dim, self.K))
        self.register_buffer("queue1_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue2", torch.randn(dim, self.K))
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue3", torch.randn(dim, self.K))
        self.register_buffer("queue3_ptr", torch.zeros(1, dtype=torch.long))
        dim = 32
        self.register_buffer("queue0_0", torch.randn(dim, self.K))
        self.register_buffer("queue0_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue1_0", torch.randn(dim, self.K))
        self.register_buffer("queue1_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue2_0", torch.randn(dim, self.K))
        self.register_buffer("queue2_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue3_0", torch.randn(dim, self.K))
        self.register_buffer("queue3_0_ptr", torch.zeros(1, dtype=torch.long))
    

    #semantic
    @torch.no_grad()
    def _dequeue_and_enqueues_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]


        if batch_size != 0:
            ptr = int(self.queues0_ptr)

            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queues0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:  
                # replace the keys at ptr (dequeue and enqueue)
                self.queues0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues0_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _dequeue_and_enqueues_1(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        
        if batch_size != 0:
            ptr = int(self.queues1_ptr)
            # assert self.K % batch_size == 0  # for simplicity

            if ptr + batch_size > self.K:
                self.queues1[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues1[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:               
                # replace the keys at ptr (dequeue and enqueue)
                self.queues1[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues1_ptr[0] = ptr
    @torch.no_grad()
    def _dequeue_and_enqueues_2(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queues2_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queues2[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues2[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queues2[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues2_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueues_3(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queues3_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queues3[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues3[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queues3[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues3_ptr[0] = ptr
            



    @torch.no_grad()
    def _dequeue_and_enqueues_0_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        if batch_size != 0:
            ptr = int(self.queues0_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queues0_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues0_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:  
                # replace the keys at ptr (dequeue and enqueue)
                self.queues0_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues0_0_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _dequeue_and_enqueues_1_0(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        
        if batch_size != 0:
            ptr = int(self.queues1_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity

            if ptr + batch_size > self.K:
                self.queues1_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues1_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:               
                # replace the keys at ptr (dequeue and enqueue)
                self.queues1_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues1_0_ptr[0] = ptr
    @torch.no_grad()
    def _dequeue_and_enqueues_2_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queues2_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queues2_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues2_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queues2_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues2_0_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueues_3_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queues3_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queues3_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queues3_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queues3_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queues3_0_ptr[0] = ptr



    #pixel
    @torch.no_grad()
    def _dequeue_and_enqueue_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # print("batch_size:", batch_size)
        if batch_size != 0:
            ptr = int(self.queue0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:  
                # replace the keys at ptr (dequeue and enqueue)
                self.queue0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue0_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue_1(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        
        if batch_size != 0:
            ptr = int(self.queue1_ptr)
            # assert self.K % batch_size == 0  # for simplicity

            if ptr + batch_size > self.K:
                self.queue1[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue1[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:               
                # replace the keys at ptr (dequeue and enqueue)
                self.queue1[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue1_ptr[0] = ptr
    @torch.no_grad()
    def _dequeue_and_enqueue_2(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue2_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue2[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue2[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue2[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue2_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue_3(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue3_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue3[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue3[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue3[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue3_ptr[0] = ptr
            



    @torch.no_grad()
    def _dequeue_and_enqueue_0_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # print("batch_size:", batch_size)
        if batch_size != 0:
            ptr = int(self.queue0_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue0_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue0_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:  
                # replace the keys at ptr (dequeue and enqueue)
                self.queue0_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue0_0_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue_1_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        
        if batch_size != 0:
            ptr = int(self.queue1_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity

            if ptr + batch_size > self.K:
                self.queue1_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue1_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:               
                # replace the keys at ptr (dequeue and enqueue)
                self.queue1_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue1_0_ptr[0] = ptr
    @torch.no_grad()
    def _dequeue_and_enqueue_2_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue2_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue2_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue2_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue2_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue2_0_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue_3_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue3_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue3_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue3_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue3_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue3_0_ptr[0] = ptr


        
    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def forward(self, x, scribble, onehot_scribble, final_selection):
        x0, x1, x2, x3, x4, x4_1, x4_2, x4_s_1, x4_s_2 = self.encoder(x)
        attention = torch.softmax(torch.einsum('nchw,cm->nhwm', x4_2, torch.cat((self.queue0_0, self.queue1_0, self.queue2_0, self.queue2_0), 1)), dim=3)
        attention1 = torch.einsum('nhwm,cm->nchw', attention, torch.cat((self.queue0, self.queue1, self.queue2, self.queue3), 1))
        x4_0 = torch.cat((x4_1, attention1), 1)
        

        #semantic
        #attention_s = torch.softmax(torch.einsum('nchw,mc->nhwm', x4_s_2, torch.cat((self.sematic_embedding_0_0, self.sematic_embedding_1_0, self.sematic_embedding_2_0, self.sematic_embedding_3_0), 0)), dim=3)
        #attention_s_1 = torch.einsum('nhwm,mc->nchw', attention_s, torch.cat((self.sematic_embedding_0, self.sematic_embedding_1, self.sematic_embedding_2, self.sematic_embedding_3), 0))
        
        attention_s = torch.softmax(torch.einsum('nchw,cm->nhwm', x4_2, torch.cat((self.queues0_0, self.queues1_0, self.queues1_0, self.queues2_0), 1)), dim=3)
        attention_s_1 = torch.einsum('nhwm,cm->nchw', attention, torch.cat((self.queues0, self.queues1, self.queues2, self.queues3), 1))

        x4_s_0 = torch.cat((x4_s_1, attention_s_1), 1)
        x4_ = x4_s_0 + x4_0
        #Pixel
        #x4_ = x4_0
        #Semantic
        #x4_ = x4_s_0 
        feature = [x0, x1, x2, x3, x4_]
        main_seg = self.main_decoder(feature)
     
        #pixel level embedding queue
        # pop the pixel embedding and calculate pixel similarity and generate map
        
        # calculate the new scibble pixel embedding and put it into the queue/4 and queue/2
        
        # get the sematic_embedding and calculate the sematic similarity and generate map
        
        # calculate the new high confident predict map and extract the enw sematic embedding and replace sematic_embedding
       
        outputs_soft = torch.softmax(main_seg, dim=1)
        feature_layer = feature[self.layer_number].permute(0,2,3,1)
        feature_layer_1 = x4_1.permute(0,2,3,1)
        feature_layer_2 = x4_2.permute(0,2,3,1)
        if scribble is None:
            pass
        else:
          
            with torch.no_grad():
                confident_score = 0.5
                scribble_position_0 = (F.interpolate(onehot_scribble[:,:,:,0].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_1 = (F.interpolate(onehot_scribble[:,:,:,1].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_2 = (F.interpolate(onehot_scribble[:,:,:,2].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_3 = (F.interpolate(onehot_scribble[:,:,:,3].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                pixel_embedding_0 = feature_layer_1[scribble_position_0] 
                pixel_embedding_1 = feature_layer_1[scribble_position_1] 
                pixel_embedding_2 = feature_layer_1[scribble_position_2] 
                pixel_embedding_3 = feature_layer_1[scribble_position_3] 
             
                self._dequeue_and_enqueue_0(pixel_embedding_0) 
                self._dequeue_and_enqueue_1(pixel_embedding_1) 
                self._dequeue_and_enqueue_2(pixel_embedding_2) 
                self._dequeue_and_enqueue_3(pixel_embedding_3) 
                
                pixel_embedding_0_0 = feature_layer_2[scribble_position_0] 
                pixel_embedding_1_0 = feature_layer_2[scribble_position_1] 
                pixel_embedding_2_0 = feature_layer_2[scribble_position_2] 
                pixel_embedding_3_0 = feature_layer_2[scribble_position_3] 
             
                self._dequeue_and_enqueue_0_0(pixel_embedding_0_0) 
                self._dequeue_and_enqueue_1_0(pixel_embedding_1_0) 
                self._dequeue_and_enqueue_2_0(pixel_embedding_2_0) 
                self._dequeue_and_enqueue_3_0(pixel_embedding_3_0) 

                # position_0 = F.interpolate(outputs_soft[:,0:1,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # position_1 = F.interpolate(outputs_soft[:,1:2,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # position_2 = F.interpolate(outputs_soft[:,2:3,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # position_3 = F.interpolate(outputs_soft[:,3:4,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)

                # final_selection_ = torch.argmax(final_selection, dim=1, keepdim=False)
                # outputs_soft_ =  torch.argmax(outputs_soft, dim=1, keepdim=False)

                # final_selection__ = final_selection_ & outputs_soft_
                # final_selection___ = F.one_hot(final_selection__, num_classes=4).permute(0,3,1,2).float()
                # new design
                # scribble_position_0 = (F.interpolate(final_selection___[:,0,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # scribble_position_1 = (F.interpolate(final_selection___[:,1,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # scribble_position_2 = (F.interpolate(final_selection___[:,2,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # scribble_position_3 = (F.interpolate(final_selection___[:,3,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # pixel_embedding_0 = feature_layer[scribble_position_0>confident_score] 
                # pixel_embedding_1 = feature_layer[scribble_position_1>confident_score] 
                # pixel_embedding_2 = feature_layer[scribble_position_2>confident_score] 
                # pixel_embedding_3 = feature_layer[scribble_position_3>confident_score] 
                
                # final_selection_0 = F.interpolate(final_selection___[:,0:1,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_1 = F.interpolate(final_selection___[:,1:2,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_2 = F.interpolate(final_selection___[:,2:3,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_3 = F.interpolate(final_selection___[:,3:4,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)

                # final_selection_0 = F.interpolate(final_selection[:,0:1,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # final_selection_1 = F.interpolate(final_selection[:,1:2,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # final_selection_2 = F.interpolate(final_selection[:,2:3,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # final_selection_3 = F.interpolate(final_selection[:,3:4,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # feature_layer = feature_layer * final_selection
                # sematic_feature_0 = (feature_layer)[final_selection_0[:,0,:,:] > confident_score]
                # sematic_feature_1 = (feature_layer)[final_selection_1[:,0,:,:]  > confident_score]
                # sematic_feature_2 = (feature_layer)[final_selection_2[:,0,:,:]  > confident_score] 
                # sematic_feature_3 = (feature_layer)[final_selection_3[:,0,:,:]  > confident_score] 
                
                # sematic_feature_0 = feature_layer[scribble_position_0]
                # sematic_feature_1 = feature_layer[scribble_position_0]
                # sematic_feature_2 = feature_layer[scribble_position_0]
                # sematic_feature_3 = feature_layer[scribble_position_0]


                sematic_mean_0 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                sematic_mean_1 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                sematic_mean_2 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                sematic_mean_3 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                #semantic 
                scribble_position_0 = (F.interpolate(onehot_scribble[:,:,:,0].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_1 = (F.interpolate(onehot_scribble[:,:,:,1].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_2 = (F.interpolate(onehot_scribble[:,:,:,2].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_3 = (F.interpolate(onehot_scribble[:,:,:,3].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]

             
                self._dequeue_and_enqueue_0(pixel_embedding_0) 
                self._dequeue_and_enqueue_1(pixel_embedding_1) 
                self._dequeue_and_enqueue_2(pixel_embedding_2) 
                self._dequeue_and_enqueue_3(pixel_embedding_3) 
                
                pixel_embedding_0_0 = feature_layer_2[scribble_position_0] 
                pixel_embedding_1_0 = feature_layer_2[scribble_position_1] 
                pixel_embedding_2_0 = feature_layer_2[scribble_position_2] 
                pixel_embedding_3_0 = feature_layer_2[scribble_position_3] 
             
                self._dequeue_and_enqueue_0_0(pixel_embedding_0_0) 
                self._dequeue_and_enqueue_1_0(pixel_embedding_1_0) 
                self._dequeue_and_enqueue_2_0(pixel_embedding_2_0) 
                self._dequeue_and_enqueue_3_0(pixel_embedding_3_0) 
                if len(pixel_embedding_0) != 0:
                    sematic_mean_0 = torch.mean(pixel_embedding_0, dim=0).unsqueeze(0) 
                if len(pixel_embedding_1) != 0:
                    sematic_mean_1 = torch.mean(pixel_embedding_1, dim=0).unsqueeze(0) 
                if len(pixel_embedding_2) != 0:
                    sematic_mean_2 = torch.mean(pixel_embedding_2, dim=0).unsqueeze(0)
                if len(pixel_embedding_3) != 0:
                    sematic_mean_3 = torch.mean(pixel_embedding_3, dim=0).unsqueeze(0)

                

                sematic_mean_0_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                sematic_mean_1_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                sematic_mean_2_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                sematic_mean_3_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                if len(pixel_embedding_0_0) != 0:
                    sematic_mean_0_0 = torch.mean(pixel_embedding_0_0, dim=0).unsqueeze(0) 
                if len(pixel_embedding_1_0) != 0:
                    sematic_mean_1_0 = torch.mean(pixel_embedding_1_0, dim=0).unsqueeze(0) 
                if len(pixel_embedding_2_0) != 0:
                    sematic_mean_2_0 = torch.mean(pixel_embedding_2_0, dim=0).unsqueeze(0)
                if len(pixel_embedding_3_0) != 0:
                    sematic_mean_3_0 = torch.mean(pixel_embedding_3_0, dim=0).unsqueeze(0)
                
                # if len(sematic_feature_0) != 0:
                #     sematic_mean_0 = torch.mean(feature_layer[position_0[:,0,:,:]  > confident_score], dim=0).unsqueeze(0) 
                # if len(sematic_feature_1) != 0:
                #     sematic_mean_1 = torch.mean(feature_layer[position_1[:,0,:,:]  > confident_score], dim=0).unsqueeze(0) 
                # if len(sematic_feature_2) != 0:
                #     sematic_mean_2 = torch.mean(feature_layer[position_2[:,0,:,:]  > confident_score], dim=0).unsqueeze(0)
                # if len(sematic_feature_3) != 0:
                #     sematic_mean_3 = torch.mean(feature_layer[position_3[:,0,:,:]  > confident_score], dim=0).unsqueeze(0)
                
                self._dequeue_and_enqueues_0(sematic_mean_0)
                self._dequeue_and_enqueues_1(sematic_mean_1)
                self._dequeue_and_enqueues_2(sematic_mean_2)
                self._dequeue_and_enqueues_3(sematic_mean_3)

                self._dequeue_and_enqueues_0_0(sematic_mean_0_0)
                self._dequeue_and_enqueues_1_0(sematic_mean_1_0)
                self._dequeue_and_enqueues_2_0(sematic_mean_2_0)
                self._dequeue_and_enqueues_3_0(sematic_mean_3_0)
                #self.sematic_embedding_0 = self.sematic_embedding_0*0.99 + sematic_mean_0*0.01
                #self.sematic_embedding_1 = self.sematic_embedding_1*0.99 + sematic_mean_1*0.01
                #self.sematic_embedding_2 = self.sematic_embedding_2*0.99 + sematic_mean_2*0.01
                #self.sematic_embedding_3 = self.sematic_embedding_3*0.99 + sematic_mean_3*0.01

                #self.sematic_embedding_0_0 = self.sematic_embedding_0_0*0.99 + sematic_mean_0_0*0.01
                #self.sematic_embedding_1_0 = self.sematic_embedding_1_0*0.99 + sematic_mean_1_0*0.01
                #self.sematic_embedding_2_0 = self.sematic_embedding_2_0*0.99 + sematic_mean_2_0*0.01
                #self.sematic_embedding_3_0 = self.sematic_embedding_3_0*0.99 + sematic_mean_3_0*0.01
            

        return main_seg, feature_layer, [self.queue0, self.queue1, self.queue2, self.queue3, self.queue0_0, self.queue1_0, self.queue2_0, self.queue3_0], [self.queues0, self.queues1, self.queues2, self.queues3, self.queues0_0, self.queues1_0, self.queues2_0, self.queues3_0]


class UNet_Matching(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Matching, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        dim = 128
        self.layer_number = 4
        
        # final
        self.K = 1000
        #self.K = 5000
        self.register_buffer('sematic_embedding_0', torch.randn(1, dim))
        self.register_buffer('sematic_embedding_1', torch.randn(1, dim))
        self.register_buffer('sematic_embedding_2', torch.randn(1, dim))
        self.register_buffer('sematic_embedding_3', torch.randn(1, dim))
        dim = 32
        self.register_buffer('sematic_embedding_0_0', torch.randn(1, dim))
        self.register_buffer('sematic_embedding_1_0', torch.randn(1, dim))
        self.register_buffer('sematic_embedding_2_0', torch.randn(1, dim))
        self.register_buffer('sematic_embedding_3_0', torch.randn(1, dim))

        # create the queue
        dim = 128
        self.register_buffer("queue0", torch.randn(dim, self.K))
        # self.queue0 = nn.functional.normalize(self.queue0, dim=0)
        self.register_buffer("queue0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue1", torch.randn(dim, self.K))
        # self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.register_buffer("queue1_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue2", torch.randn(dim, self.K))
        # self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue3", torch.randn(dim, self.K))
        # self.queue3 = nn.functional.normalize(self.queue3, dim=0)
        self.register_buffer("queue3_ptr", torch.zeros(1, dtype=torch.long))
        dim = 32
        self.register_buffer("queue0_0", torch.randn(dim, self.K))
        # self.queue0 = nn.functional.normalize(self.queue0, dim=0)
        self.register_buffer("queue0_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue1_0", torch.randn(dim, self.K))
        # self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.register_buffer("queue1_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue2_0", torch.randn(dim, self.K))
        # self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_0_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue3_0", torch.randn(dim, self.K))
        # self.queue3 = nn.functional.normalize(self.queue3, dim=0)
        self.register_buffer("queue3_0_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _dequeue_and_enqueue_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        if batch_size != 0:
            ptr = int(self.queue0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:  
                # replace the keys at ptr (dequeue and enqueue)
                self.queue0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue0_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue_1(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        
        if batch_size != 0:
            ptr = int(self.queue1_ptr)
            # assert self.K % batch_size == 0  # for simplicity

            if ptr + batch_size > self.K:
                self.queue1[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue1[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:               
                # replace the keys at ptr (dequeue and enqueue)
                self.queue1[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue1_ptr[0] = ptr
    @torch.no_grad()
    def _dequeue_and_enqueue_2(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue2_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue2[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue2[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue2[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue2_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue_3(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue3_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue3[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue3[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue3[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue3_ptr[0] = ptr
            



    @torch.no_grad()
    def _dequeue_and_enqueue_0_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # print("batch_size:", batch_size)
        if batch_size != 0:
            ptr = int(self.queue0_0_ptr)

            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue0_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue0_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:  
                # replace the keys at ptr (dequeue and enqueue)
                self.queue0_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue0_0_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue_1_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        
        if batch_size != 0:
            ptr = int(self.queue1_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity

            if ptr + batch_size > self.K:
                self.queue1_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue1_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:               
                # replace the keys at ptr (dequeue and enqueue)
                self.queue1_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue1_0_ptr[0] = ptr
    @torch.no_grad()
    def _dequeue_and_enqueue_2_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue2_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue2_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue2_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue2_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue2_0_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue_3_0(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if batch_size != 0:
            ptr = int(self.queue3_0_ptr)
            # assert self.K % batch_size == 0  # for simplicity
            if ptr + batch_size > self.K:
                self.queue3_0[:, ptr: self.K] = keys.T[:, :self.K - ptr]
                self.queue3_0[:, : (ptr + batch_size) % self.K] = keys.T[:, self.K - ptr:]
            else:
                # replace the keys at ptr (dequeue and enqueue)
                self.queue3_0[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue3_0_ptr[0] = ptr


        
    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def forward(self, x, scribble, onehot_scribble, final_selection):
        x0, x1, x2, x3, x4, x4_1, x4_2, x4_s_1, x4_s_2 = self.encoder(x)

        
        #orginal 
        attention = torch.softmax(torch.einsum('nchw,cm->nhwm', x4_2, torch.cat((self.queue0_0, self.queue1_0, self.queue2_0, self.queue3_0), 1)), dim=3)
        attention1 = torch.einsum('nhwm,cm->nchw', attention, torch.cat((self.queue0, self.queue1, self.queue2, self.queue3), 1))
        x4_0 = torch.cat((x4_1, attention1), 1)
     

        #attention = torch.softmax(torch.einsum('nchw,cm->nhwm', x4_2, torch.cat((self.queue0_0, self.queue1_0, self.queue2_0, self.queue2_0), 1)), dim=3)
        #max_idx = torch.argmax(attention, dim=3)
        #out = torch.zeros_like(attention)
        #out.scatter_(3, max_idx.unsqueeze(3), 1)
        #attention1 = torch.einsum('nhwm,cm->nchw', out, torch.cat((self.queue0, self.queue1, self.queue2, self.queue3), 1))
        #x4_0 = torch.cat((x4_1, attention1), 1)
        
        #orginal
        attention_s = torch.softmax(torch.einsum('nchw,mc->nhwm', x4_s_2, torch.cat((self.sematic_embedding_0_0, self.sematic_embedding_1_0, self.sematic_embedding_2_0, self.sematic_embedding_3_0), 0)), dim=3)
        attention_s_1 = torch.einsum('nhwm,mc->nchw', attention_s, torch.cat((self.sematic_embedding_0, self.sematic_embedding_1, self.sematic_embedding_2, self.sematic_embedding_3), 0))
        x4_s_0 = torch.cat((x4_s_1, attention_s_1), 1)
        
        #hard
        #attention_s = torch.softmax(torch.einsum('nchw,mc->nhwm', x4_s_2, torch.cat((self.sematic_embedding_0_0, self.sematic_embedding_1_0, self.sematic_embedding_2_0, self.sematic_embedding_3_0), 0)), dim=3)
        #max_idx_s = torch.argmax(attention_s, dim=3)
        #out_s = torch.zeros_like(attention_s)
        #out_s.scatter_(3, max_idx_s.unsqueeze(3), 1)
        #attention_s_1 = torch.einsum('nhwm,mc->nchw', out_s, torch.cat((self.sematic_embedding_0, self.sematic_embedding_1, self.sematic_embedding_2, self.sematic_embedding_3), 0))
        #x4_s_0 = torch.cat((x4_s_1, attention_s_1), 1)

        x4_ = x4_s_0 + x4_0
        #Pixel
        #x4_ = x4_0
        #Semantic
        #x4_ = x4_s_0 
        feature = [x0, x1, x2, x3, x4_]
        #feature = [x0, x1, x2, x3, x4]

        # feature = [Dropout(i) for i in feature]
        main_seg = self.main_decoder(feature)
     
        #pixel level embedding queue
        # pop the pixel embedding and calculate pixel similarity and generate map
        
        # calculate the new scibble pixel embedding and put it into the queue/4 and queue/2
        
        # get the sematic_embedding and calculate the sematic similarity and generate map
        
        # calculate the new high confident predict map and extract the enw sematic embedding and replace sematic_embedding
       
        outputs_soft = torch.softmax(main_seg, dim=1)

        feature_layer = feature[self.layer_number].permute(0,2,3,1)
        feature_layer_1 = x4_1.permute(0,2,3,1)
        feature_layer_2 = x4_2.permute(0,2,3,1)
        if scribble is None:
            pass
        else:
          
            with torch.no_grad():
                confident_score = 0.5
                
                scribble_position_0 = (F.interpolate(onehot_scribble[:,:,:,0].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_1 = (F.interpolate(onehot_scribble[:,:,:,1].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_2 = (F.interpolate(onehot_scribble[:,:,:,2].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                scribble_position_3 = (F.interpolate(onehot_scribble[:,:,:,3].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]

                pixel_embedding_0 = feature_layer_1[scribble_position_0] 
                pixel_embedding_1 = feature_layer_1[scribble_position_1] 
                pixel_embedding_2 = feature_layer_1[scribble_position_2] 
                pixel_embedding_3 = feature_layer_1[scribble_position_3] 
             
                self._dequeue_and_enqueue_0(pixel_embedding_0) 
                self._dequeue_and_enqueue_1(pixel_embedding_1) 
                self._dequeue_and_enqueue_2(pixel_embedding_2) 
                self._dequeue_and_enqueue_3(pixel_embedding_3) 
                
                pixel_embedding_0_0 = feature_layer_2[scribble_position_0] 
                pixel_embedding_1_0 = feature_layer_2[scribble_position_1] 
                pixel_embedding_2_0 = feature_layer_2[scribble_position_2] 
                pixel_embedding_3_0 = feature_layer_2[scribble_position_3] 
             
                self._dequeue_and_enqueue_0_0(pixel_embedding_0_0) 
                self._dequeue_and_enqueue_1_0(pixel_embedding_1_0) 
                self._dequeue_and_enqueue_2_0(pixel_embedding_2_0) 
                self._dequeue_and_enqueue_3_0(pixel_embedding_3_0) 

                # position_0 = F.interpolate(outputs_soft[:,0:1,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # position_1 = F.interpolate(outputs_soft[:,1:2,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # position_2 = F.interpolate(outputs_soft[:,2:3,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # position_3 = F.interpolate(outputs_soft[:,3:4,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_ = torch.argmax(final_selection, dim=1, keepdim=False)
                # outputs_soft_ =  torch.argmax(outputs_soft, dim=1, keepdim=False)
                # final_selection__ = final_selection_ & outputs_soft_
                # final_selection___ = F.one_hot(final_selection__, num_classes=4).permute(0,3,1,2).float()
                # new design
                # scribble_position_0 = (F.interpolate(final_selection___[:,0,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # scribble_position_1 = (F.interpolate(final_selection___[:,1,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # scribble_position_2 = (F.interpolate(final_selection___[:,2,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # scribble_position_3 = (F.interpolate(final_selection___[:,3,:,:].unsqueeze(1), size=feature_layer_1.shape[1:3], mode='bilinear', align_corners=True) ==1)[:,0,:,:]
                # print('scribble_position_0:',torch.unique(scribble_position_0, return_counts=True))
                # print('scribble_position_1:',torch.unique(scribble_position_1, return_counts=True))
                # print('scribble_position_2:',torch.unique(scribble_position_2, return_counts=True))
                # print('scribble_position_3:',torch.unique(scribble_position_3, return_counts=True))
                # pixel_embedding_0 = feature_layer[scribble_position_0>confident_score] 
                # pixel_embedding_1 = feature_layer[scribble_position_1>confident_score] 
                # pixel_embedding_2 = feature_layer[scribble_position_2>confident_score] 
                # pixel_embedding_3 = feature_layer[scribble_position_3>confident_score] 
                
                # final_selection_0 = F.interpolate(final_selection___[:,0:1,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_1 = F.interpolate(final_selection___[:,1:2,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_2 = F.interpolate(final_selection___[:,2:3,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
                # final_selection_3 = F.interpolate(final_selection___[:,3:4,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True)
               


                # final_selection_0 = F.interpolate(final_selection[:,0:1,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # final_selection_1 = F.interpolate(final_selection[:,1:2,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # final_selection_2 = F.interpolate(final_selection[:,2:3,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # final_selection_3 = F.interpolate(final_selection[:,3:4,:,:], size=feature_layer.shape[1:3], mode='bilinear', align_corners=True).permute(0,2,3,1)
                # feature_layer = feature_layer * final_selection
                # sematic_feature_0 = (feature_layer)[final_selection_0[:,0,:,:] > confident_score]
                # sematic_feature_1 = (feature_layer)[final_selection_1[:,0,:,:]  > confident_score]
                # sematic_feature_2 = (feature_layer)[final_selection_2[:,0,:,:]  > confident_score] 
                # sematic_feature_3 = (feature_layer)[final_selection_3[:,0,:,:]  > confident_score] 
                
                # sematic_feature_0 = feature_layer[scribble_position_0]
                # sematic_feature_1 = feature_layer[scribble_position_0]
                # sematic_feature_2 = feature_layer[scribble_position_0]
                # sematic_feature_3 = feature_layer[scribble_position_0]
                sematic_mean_0 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                sematic_mean_1 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                sematic_mean_2 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                sematic_mean_3 = torch.zeros(1, feature_layer_1.shape[-1]).cuda()
                
                if len(pixel_embedding_0) != 0:
                    sematic_mean_0 = torch.mean(pixel_embedding_0, dim=0).unsqueeze(0) 
                if len(pixel_embedding_1) != 0:
                    sematic_mean_1 = torch.mean(pixel_embedding_1, dim=0).unsqueeze(0) 
                if len(pixel_embedding_2) != 0:
                    sematic_mean_2 = torch.mean(pixel_embedding_2, dim=0).unsqueeze(0)
                if len(pixel_embedding_3) != 0:
                    sematic_mean_3 = torch.mean(pixel_embedding_3, dim=0).unsqueeze(0)

                sematic_mean_0_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                sematic_mean_1_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                sematic_mean_2_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                sematic_mean_3_0 = torch.zeros(1, feature_layer_2.shape[-1]).cuda()
                if len(pixel_embedding_0_0) != 0:
                    sematic_mean_0_0 = torch.mean(pixel_embedding_0_0, dim=0).unsqueeze(0) 
                if len(pixel_embedding_1_0) != 0:
                    sematic_mean_1_0 = torch.mean(pixel_embedding_1_0, dim=0).unsqueeze(0) 
                if len(pixel_embedding_2_0) != 0:
                    sematic_mean_2_0 = torch.mean(pixel_embedding_2_0, dim=0).unsqueeze(0)
                if len(pixel_embedding_3_0) != 0:
                    sematic_mean_3_0 = torch.mean(pixel_embedding_3_0, dim=0).unsqueeze(0)
                
                # if len(sematic_feature_0) != 0:
                #     sematic_mean_0 = torch.mean(feature_layer[position_0[:,0,:,:]  > confident_score], dim=0).unsqueeze(0) 
                # if len(sematic_feature_1) != 0:
                #     sematic_mean_1 = torch.mean(feature_layer[position_1[:,0,:,:]  > confident_score], dim=0).unsqueeze(0) 
                # if len(sematic_feature_2) != 0:
                #     sematic_mean_2 = torch.mean(feature_layer[position_2[:,0,:,:]  > confident_score], dim=0).unsqueeze(0)
                # if len(sematic_feature_3) != 0:
                #     sematic_mean_3 = torch.mean(feature_layer[position_3[:,0,:,:]  > confident_score], dim=0).unsqueeze(0)
                
            
                self.sematic_embedding_0 = self.sematic_embedding_0*0.99 + sematic_mean_0*0.01
                self.sematic_embedding_1 = self.sematic_embedding_1*0.99 + sematic_mean_1*0.01
                self.sematic_embedding_2 = self.sematic_embedding_2*0.99 + sematic_mean_2*0.01
                self.sematic_embedding_3 = self.sematic_embedding_3*0.99 + sematic_mean_3*0.01

                self.sematic_embedding_0_0 = self.sematic_embedding_0_0*0.99 + sematic_mean_0_0*0.01
                self.sematic_embedding_1_0 = self.sematic_embedding_1_0*0.99 + sematic_mean_1_0*0.01
                self.sematic_embedding_2_0 = self.sematic_embedding_2_0*0.99 + sematic_mean_2_0*0.01
                self.sematic_embedding_3_0 = self.sematic_embedding_3_0*0.99 + sematic_mean_3_0*0.01
            
            
           
        return main_seg, feature_layer, [self.queue0, self.queue1, self.queue2, self.queue3, self.queue0_0, self.queue1_0, self.queue2_0, self.queue3_0], [self.sematic_embedding_0, self.sematic_embedding_1,self.sematic_embedding_2,self.sematic_embedding_3, self.sematic_embedding_0_0, self.sematic_embedding_1_0, self.sematic_embedding_2_0, self.sematic_embedding_3_0]
       
class UNet_Trans(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Trans, self).__init__()
        # print("in_chns:",in_chns)
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        transformer_params = {'dim': 1024,
                  'depth': 12,
                  'heads': 8,
                  'dim_head': 32,
                  'mlp_dim': 2048,
                  'num_classes': 4,
                  "num_patches": 64,
                  "pool": 'cls'}

        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.transformer_decoder = Transformer_Decoder(transformer_params)

    def forward(self, x, superpixel):

        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        
        classification_result = self.transformer_decoder(x, feature, superpixel)
        
        return main_seg, classification_result




class UNet_CCT_3H(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT_3H, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [FeatureNoise()(i) for i in feature]
        aux_seg2 = self.aux_decoder1(aux2_feature)
        return main_seg, aux_seg1, aux_seg2
