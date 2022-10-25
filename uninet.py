
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import zeros_
import torch.utils.checkpoint as checkpoint

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from model_utils import _make_divisible, SqueezeExcitation, to3d, to4d


class LocalDSM(nn.Module):
    def __init__(self, in_features, out_features, stride=1, mlp_ratio=4, head_dim=32,
                 qkv_bias=True, qk_scale=None, drop=0., drop_path=0., attn_drop=0., seq_l=196):
        super().__init__()
        h = w = int(seq_l ** 0.5)
        new_h = new_w = math.ceil(h / stride)
        self.h = h
        self.new_h = new_h
        self.new_N = self.new_h * self.new_h
        if stride == 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
        else:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=(3, 3), padding=(1, 1),
                        groups=in_features, stride=stride, bias=True),
            nn.BatchNorm2d(in_features),
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=True),
        )
        
    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 3:
            x = to4d(x)
        return self.downsample(x) + self.residual(x)


class LocalGlobalDSM(nn.Module):
    def __init__(self, in_features, out_features, stride=1, mlp_ratio=4, head_dim=32,
                 qkv_bias=True, qk_scale=None, drop=0., drop_path=0., attn_drop=0., seq_l=196):
        super().__init__()
        out_dim = out_features or in_features
        self.num_heads = out_features // head_dim
        self.head_dim = head_dim
        self.out_features = out_features
        self.scale = qk_scale or head_dim ** -0.5

        h = w = int(seq_l ** 0.5)
        new_h = new_w = math.ceil(h / stride)
        self.h = h
        self.new_h = new_h
        self.new_N = self.new_h * self.new_h

        if stride == 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
        else:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )

        self.q = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=(3, 3), padding=(1, 1),
                        groups=in_features, stride=stride, bias=True),
            nn.BatchNorm2d(in_features),
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=True),
        )

        self.q_norm = nn.LayerNorm(out_features)
        self.kv_norm = nn.LayerNorm(in_features)
        self.kv = nn.Linear(in_features, out_features * 2, bias=qkv_bias)
        self.proj = nn.Linear(out_features, out_features)

    def forward(self, x):
        x_shape = x.shape
        if len(x.shape) == 3:
            B, N, C = x_shape
        else:
            B, C, H, W = x_shape
            N = H * W
        
        x = to4d(x)
        residual = to3d(self.residual(x))
        q = to3d(self.q(x))
        x = to3d(x)

        q = self.q_norm(q)
        x = self.kv_norm(x)
        q = q.reshape(B, self.new_N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, self.new_N, self.out_features)

        x = self.proj(x)
        return x + residual


class DWConvBlock(nn.Module):
    def __init__(self, in_features, out_features=None, stride=1,
                 mlp_ratio=4, use_se=True, drop=0., drop_path=0.,
                 seq_l=196, head_dim=32, init_values=1e-6, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        if in_features != out_features or stride != 1:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
        else:
            self.residual = nn.Identity()

        self.b1 = None
        if in_features != hidden_features or stride != 1:
            layers_b1 = []
            layers_b1.append(nn.BatchNorm2d(in_features))
            layers_b1.append(nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1),
                                       stride=1, padding=(0, 0), bias=False))
            layers_b1.append(nn.BatchNorm2d(hidden_features))
            layers_b1.append(nn.GELU())
            self.b1 = nn.Sequential(*layers_b1)

        layers = []
        layers.append(nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), padding=(1, 1),
                                groups=hidden_features, stride=stride, bias=False))
        layers.append(nn.BatchNorm2d(hidden_features))
        layers.append(nn.GELU())
        if use_se:
            layers.append(SqueezeExcitation(hidden_features))

        layers.append(nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1), padding=(0, 0)))
        layers.append(nn.BatchNorm2d(out_features))
        self.b2 = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if init_values != -1:
            zeros_(self.b2[-1].weight)

    def forward(self, x):
        residual = self.residual(x)
        if self.b1 is not None:
            x = self.b1(x)
        x = self.b2(x)

        return residual + self.drop_path(x)


class Attention(nn.Module):
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 seq_l=196, fp32_attn=False):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.custom_flops = 2 * seq_l * seq_l * dim
        self.fp32_attn = fp32_attn

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if self.fp32_attn:
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        else:
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.fp32_attn:
            x = x.to(self.proj.weight.dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, in_features, out_features, stride=1, mlp_ratio=4, head_dim=32,
                 qkv_bias=True, qk_scale=None, drop=0., drop_path=0., attn_drop=0., seq_l=196,
                 init_values=1e-6, fp32_attn=False,):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.stride = stride
        self.in_features = in_features
        self.out_features = out_features
        mlp_hidden_dim = int(in_features * mlp_ratio)
        num_heads = in_features // head_dim
        self.init_values = init_values
        self.pos_embed = nn.Conv2d(in_features, in_features, 3, padding=1, groups=in_features)
        self.attn = Attention(in_features, out_features, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq_l=seq_l,
                              fp32_attn=fp32_attn)
        if stride != 1 or in_features != out_features:
            self.ds = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
        else:
            self.norm2 = nn.LayerNorm(in_features)
            self.mlp = Mlp(in_features=in_features, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
                self.gamma_2 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, head=0, mask_type=None):
        x = x + to3d(self.pos_embed(to4d(x)))
        if self.stride == 1 and self.in_features == self.out_features:
            if self.init_values != -1:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), head=head, mask_type=mask_type))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), head=head, mask_type=mask_type))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            residual = to3d(self.residual(to4d(x)))
            x = self.norm1(x)
            x = to3d(self.ds(to4d(x)))
            x = self.attn(x)
            if self.init_values != -1:
                x = residual + self.gamma_1 * x
            else:
                x = residual + x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, repeats, expansion, channels, strides=[1, 2, 2, 2, 1, 2], num_classes=1000, drop_path_rate=0.,
                 input_size=224, weight_init='', head_dim=32, final_head_dim=1280, final_drop=0.0, init_values=1e-6, 
                 block_ops=[DWConvBlock] * 3 + [AttentionBlock] * 3, checkpoint=0, stem_dim=32, 
                 ds_ops=[LocalDSM] * 3 + [LocalGlobalDSM] * 2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.checkpoint = checkpoint

        # stem_dim = 32
        h = w = input_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(repeats))]  # stochastic depth decay rule
        dpr.reverse()

        cin = stem_dim
        blocks = []
        h = w = math.ceil(h / 2)
        seq_l = h * w
        for stage in range(len(strides)):
            cout = channels[stage]
            block_op = block_ops[stage]
            # print(f'stage {stage}, cin {cin}, cout {cout}, s {strides[stage]}, e {expansion[stage]} b {block_op}')

            if stage != 0:
                blocks.append(ds_ops[stage - 1](cin, cout, stride=strides[stage], seq_l=seq_l, head_dim=head_dim))
                h = w = math.ceil(h / strides[stage])
                seq_l = h * w
                cin = cout

            # cin = cout
            for i in range(repeats[stage]):
                stride = strides[stage] if i == 0 else 1
                blocks.append(block_op(cin, cout, stride=1, mlp_ratio=expansion[stage],
                                       drop_path=dpr.pop(), seq_l=seq_l, head_dim=head_dim,
                                       init_values=init_values))
                cin = cout
        self.blocks = nn.Sequential(*blocks)

        head_dim = final_head_dim
        self.head = nn.Sequential(
            nn.Conv2d(cout, head_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(head_dim),
            nn.GELU(),
        )
        self.final_drop = nn.Dropout(final_drop) if final_drop > 0.0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_dim, num_classes)

        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            # trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'dist_token'}

    def forward_features(self, x):

        x = self.stem(x)
        for i, blk in enumerate(self.blocks):
            if isinstance(blk, DWConvBlock):
                x = to4d(x)
            if isinstance(blk, AttentionBlock):
                x = to3d(x)
            if i < self.checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = to4d(x)
        x = self.head(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_drop(x)
        x = self.classifier(x)
        return x


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


@register_model
def UniNetB0(**kwargs): # 11.451M, 0.555G, 160
    repeats = [1, 2, 4, 4, 4, 8]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [24, 48, 80, 128, 128, 256]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 160

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

@register_model
def UniNetB1(**kwargs):  # 11.451M, 1.118G, 224
    repeats = [1, 2, 4, 4, 4, 8]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [24, 48, 80, 128, 128, 256]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 224

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

@register_model
def UniNetB2(**kwargs):  # 16.211M, 2.159G, 256
    repeats = [2, 3, 6, 6, 6, 12]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [24, 48, 80, 128, 128, 256]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 256

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

@register_model
def UniNetB3(**kwargs):  # 24.02M, 4.258G, 288
    repeats = [2, 3, 7, 7, 7, 14]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [24, 56, 96, 160, 160, 288]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 288

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


@register_model
def UniNetB4(**kwargs):  # 43.796M, 9.429G, 320
    repeats = [2, 4, 9, 9, 9, 18]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [32, 64, 112, 192, 192, 352]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 320

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

@register_model
def UniNetB5(**kwargs):  # 72.883M, *, 320
    repeats = [3, 5, 10, 10, 10, 20]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [32, 64, 112, 224, 224, 448]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 384

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

@register_model
def UniNetB6(**kwargs):  # 117M, *, 320
    repeats = [4, 6, 12, 12, 12, 24]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [48, 96, 160, 256, 256, 512]
    final_drop = 0.0
    block_ops = [DWConvBlock] * 4 + [AttentionBlock] * 2
    input_size = 448

    print(f'channels {channels}, repeats {repeats}, expansion {expansion}, block_ops {block_ops}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, 
                        input_size=input_size, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model
