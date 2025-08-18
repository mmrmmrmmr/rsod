import torch.nn as nn
import torch
from einops import rearrange
# from torchsummary import summary
# from SLaK import Block
try:
    from .SLaK import Block
    from .FFCA import *
except:
    from SLaK import Block
    from FFCA import *
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, MaxPool2d,
                      build_norm_layer)
from mmdet.models.layers.csp_layer import \
    DarknetBottleneck as MMDET_DarknetBottleneck
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from torch import Tensor

# from mmyolo.registry import MODELS


class DarknetBottleneck(MMDET_DarknetBottleneck):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = True,
                 use_depthwise: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = conv(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

class myCSP(nn.modules):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int = 1,
            expand_ratio: float = 0.5,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            k=0.5,
            if_cat=False) -> None:
        super().__init__(init_cfg=init_cfg)
        self.if_cat = if_cat
        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.if_cat:
            self.final_conv = ConvModule(
                (2 + num_blocks) * self.mid_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.final_conv = DepthwiseSeparableConvModule(
                (1 + num_blocks) * self.mid_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks))
        self.se = SELayer(2 * self.mid_channels)
        
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        bs, _, _, _ = x_main.size()
        # x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        y = self.se(x_main)
        # return y
        _, indices_1 = y.topk(dim=1,k=self.mid_channels,largest=True)
        indices_1 = torch.squeeze(indices_1)
        # print(indices_1.shape)
        _, indices_2 = y.topk(dim=1,k=self.mid_channels,largest=False)
        indices_2 = torch.squeeze(indices_2)

        indices_1 = rearrange(indices_1,'b c -> (b c)')
        indices_2 = rearrange(indices_2,'b c -> (b c)')
        # return indices_2, x_main
        x_main = rearrange(x_main, 'b c h w -> (b c) h w')
        # return indices_1, x_main
        x_1 = torch.index_select(x_main,0,indices_1)
        x_2 = torch.index_select(x_main,0,indices_2)
        x_1 = rearrange(x_1, '(b c) h w -> b c h w',b=bs)
        x_2 = rearrange(x_2, '(b c) h w -> b c h w',b=bs)
        # return indices_1, x_1
        # x_main = list((x_1, x_2))
        if self.if_cat == False:
            x_main = [x_2]
        x_main.extend(blocks(x_1) for blocks in self.blocks)
        # return x_1
        return self.final_conv(torch.cat(x_main, 1))
    
    
class myCSP2(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            k=0.5,
            if_cat=False) -> None:
        super().__init__(init_cfg=init_cfg)
        self.if_cat = if_cat
        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            (1 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mid_conv = ConvModule(
            self.mid_channels,
            self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks))
        self.se = SELayer(2 * self.mid_channels)
        
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        bs, _, _, _ = x_main.size()
        # x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        y = self.se(x_main)
        # return y
        _, indices_1 = y.topk(dim=1,k=self.mid_channels,largest=True)
        _, indices_2 = y.topk(dim=1,k=self.mid_channels,largest=False)
        indices_1 = rearrange(indices_1,'b c -> (b c)')
        indices_2 = rearrange(indices_2,'b c -> (b c)')
        # return indices_2, x_main
        x_main = rearrange(x_main, 'b c h w -> (b c) h w')
        # return indices_1, x_main
        x_1 = torch.index_select(x_main,0,indices_1)
        x_2 = torch.index_select(x_main,0,indices_2)
        x_1 = rearrange(x_1, '(b c) h w -> b c h w',b=bs)
        x_2 = rearrange(x_2, '(b c) h w -> b c h w',b=bs)
        # return indices_1, x_1
        # x_main = list((x_1, x_2))
        # x_main = [x_2]
        x_main = [self.mid_conv(x_2)]
        x_main.extend(blocks(x_1) for blocks in self.blocks)
        # return x_1
        return self.final_conv(torch.cat(x_main, 1))
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        # print(y.size())
        # print(self.fc)
        y = self.fc(y)
        y = y.view(b, c, 1, 1) #对应Excitation操作
        # return x * y.expand_as(x)
        # return y.squeeze(2).squeeze(2)
        return y


class cspp(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, 
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            num=2
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.ModuleList()
        self.conv_begin = DepthwiseSeparableConvModule(
                in_channels,
                in_channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=None)
        for _ in range(num):
            self.conv.append(
                DepthwiseSeparableConvModule(
                in_channels,
                in_channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=None)
            )
        self.conv_f = DepthwiseSeparableConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, x):
        x = self.conv_begin(x)
        z = x
        for i in range(len(self.conv)): 
            x = self.conv[i](x)
            z = x + z
        z = self.conv_f(z)
        return z
    
class mybottle(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, 
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_ch = in_channels
        self.convc = lspp(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg
        )
        self.soft = nn.Softmax2d()
        self.qk = DepthwiseSeparableConvModule(
            in_channels,
            2*in_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.convf = DepthwiseSeparableConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


    def forward(self, x):
        x1 = self.convc(x)
        x2 = self.qk(x)
        q, k = list(x2.split((self.in_ch, self.in_ch), 1))
        att = self.soft(torch.mul(q,k))
        # z = torch.cat([x1,torch.mul(x1,att)],1)
        z = x1 + torch.mul(x1,att)
        z = self.convf(z)
        return z

class mybottle2(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, 
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_ch = in_channels
        self.convc = lspp(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg
        )
        self.gl = EfficientAttention(out_channels)
        # self.gl = nn.Identity(out_channels)


    def forward(self, x):
        x1 = self.convc(x)
        x2 = self.gl(x1)
        # z = torch.cat([x1,torch.mul(x1,att)],1)
        z = x + x2
        return z

class myCSP_lspp(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.ModuleList(
            mybottle(
                self.mid_channels,
                self.mid_channels,
                conv_cfg,
                norm_cfg,
                act_cfg,
                init_cfg
            ) for _ in range(num_blocks))
        
        self.se = SELayer(self.mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))

class lspp(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, 
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            size=[5,9,13]
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.ModuleList()
        self.conv_begin = DepthwiseSeparableConvModule(
                in_channels,
                in_channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=None)
        for i in size:
            self.conv.append(
                Block(
                dim=in_channels,
                kernel_size=(i,i))
            )
        self.conv_f = DepthwiseSeparableConvModule(
            (len(size)+1)*in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # self.se = SELayer(4 * in_channels)
        # self.mid = 2 * in_channels
        
    def forward(self, x):
        z = self.conv_begin(x)
        z = [z]
        for i in range(len(self.conv)): 
            y = self.conv[i](x)
            z.append(y)
        z = torch.cat(z, dim=1)
        # print(z.shape)
        # bs, _, _, _ = z.size()
        # # x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        # y = self.se(z)
        # # return y
        # _, indices_1 = y.squeeze(2).squeeze(2).topk(dim=1,k=self.mid,largest=True)
        # indices_1 = rearrange(indices_1,'b c -> (b c)')
        # # return indices_2, x_main
        # z = rearrange(z, 'b c h w -> (b c) h w')
        # y = rearrange(y, 'b c h w -> (b c) h w')

        # # return indices_1, x_main
        # x_1 = torch.index_select(z,0,indices_1)
        # y = torch.index_select(y,0,indices_1)

        # z = rearrange(x_1*y.expand_as(x_1), '(b c) h w -> b c h w',b=bs)
        z = self.conv_f(z)
        return z

class lspp_down(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, 
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            size=[5,9,13]
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.ModuleList()
        self.conv_begin = DepthwiseSeparableConvModule(
                in_channels,
                in_channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=None)
        for i in size:
            self.conv.append(
                Block(
                dim=in_channels,
                kernel_size=(i,i))
            )
        self.mid = 2 * out_channels
        self.conv_f = DepthwiseSeparableConvModule(
            self.mid,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.se = SELayer((len(size)+1) * in_channels)
        
    def forward(self, x):
        z = self.conv_begin(x)
        z = [z]
        for i in range(len(self.conv)): 
            y = self.conv[i](x)
            z.append(y)
        z = torch.cat(z, dim=1)
        # print(z.shape)
        bs, _, _, _ = z.size()
        # x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        y = self.se(z)
        # return y
        _, indices_1 = y.squeeze(2).squeeze(2).topk(dim=1,k=self.mid,largest=True)
        indices_1 = rearrange(indices_1,'b c -> (b c)')
        # return indices_2, x_main
        z = rearrange(z, 'b c h w -> (b c) h w')
        y = rearrange(y, 'b c h w -> (b c) h w')

        # return indices_1, x_main
        x_1 = torch.index_select(z,0,indices_1)
        y = torch.index_select(y,0,indices_1)

        z = rearrange(x_1*y.expand_as(x_1), '(b c) h w -> b c h w',b=bs)
        z = self.conv_f(z)
        return z


class myCSP_lspp2(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.ModuleList(
            mybottle2(
                self.mid_channels,
                self.mid_channels,
                conv_cfg,
                norm_cfg,
                act_cfg,
                init_cfg
            ) for _ in range(num_blocks))
        
        # self.se = SELayer(self.mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
 
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
 
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                         )
    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=4, group_split=[0, 4], kernel_sizes=[5], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes

        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2d(dim, 3*group_head*self.dim_head, 1, 1, 0, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size!=1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x) #(b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous() #(3 b (m d) h w)
        q, k, v = qkv #(b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v) #(b (m d) h w)
        return res

    def low_fre_attention(self, x : torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        kv = avgpool(x) #(b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h*w)//(self.window_size**2)).permute(1, 0, 2, 4, 3).contiguous() #(2 b m (H W) d)
        k, v = kv #(b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2) #(b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v #(b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))

# if __name__ == '__main__':
    # x = torch.rand(2,192,640,640)
    # mm = EfficientAttention(256)
    # mm = myCSP(256,64)

    # # i, y = mm(x)
    # # blocks = DarknetBottleneck(
    # #             192,
    # #             192,
    # #             expansion=1,
    # #             kernel_size=(3, 3),
    # #             padding=(1, 1),
    # #             add_identity=True,
    # #             use_depthwise=False)
    # x = torch.rand(2,256,80,80)
    # y = mm(x)

