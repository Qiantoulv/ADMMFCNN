import copy
import math

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2





class ChannelAttention(nn.Module):
    def __init__(self, frequency_bands, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(frequency_bands, frequency_bands // reduction, bias=False)
        self.fc2 = nn.Linear(frequency_bands // reduction, frequency_bands, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()  # b: batch_size, c: frequency_bands, h: channel_num, w: timepoint
        y = x.mean(dim=(2, 3))
        y = self.fc1(y)
        y = nn.ReLU()(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  

class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.clamp(x.var(dim=self.dim), 1e-6, 1e6)
        return torch.log(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i * self.stride
            input = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)

        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])



class LightweightConv1d(nn.Module):
    """
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding=0, num_heads=1, weight_softmax=False,
                 bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, -1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output


# Data shape: batch * filterBand * chan * time
class Mul(nn.Module):
    def __init__(self, num_classes=4,num_bands=9, embed_dim=32,
                 win_len=250,):
        super().__init__()

        self.win_len = win_len
        self.dropout = nn.Dropout()

        self.frequencyattn=ChannelAttention(num_bands)
        self.temp_conv1 = nn.Conv2d(22, embed_dim // 4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(22, embed_dim // 4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(22, embed_dim // 4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(22, embed_dim // 4, (1, 65), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (9, 1))

        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()

        self.var_pool = VarPoold(50, 15)
        self.avg_pool = nn.AvgPool1d(50, 15)


        self.attn=TransformerEncoder(6, 32)

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(64, 64, (2, 1)),
            nn.BatchNorm2d(64),
            nn.ELU()
        )


        self.classify = nn.Linear(embed_dim*64, num_classes)

    def forward(self, x):
        x   = self.frequencyattn(x)
        x=x.transpose(1,2)
        print(x.shape)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = x.squeeze(dim=2)

        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')
        x1=self.attn(x1)
        x2=self.attn(x2)
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)
        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)

        x = x.reshape(x.size(0), -1)
        out = self.classify(x)


        return out



