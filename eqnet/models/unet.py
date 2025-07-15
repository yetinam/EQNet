# U-Net implementation adapted from: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def moving_normalize(data, filter=1024, stride=128):
    nb, nch, nx, nt = data.shape

    # if nt % stride == 0:
    #     pad = max(filter - stride, 0)
    # else:
    #     pad = max(filter - (nt % stride), 0)
    # pad1 = pad // 2
    # pad2 = pad - pad1
    padding = filter // 2

    with torch.no_grad():
        # data_ = F.pad(data, (pad1, pad2, 0, 0), mode="reflect")
        data_ = F.pad(data, (padding, padding, 0, 0), mode="reflect")
        mean = F.avg_pool2d(data_, kernel_size=(1, filter), stride=(1, stride))
        mean = F.interpolate(mean, scale_factor=(1, stride), mode="bilinear", align_corners=False)[:, :, :nx, :nt]
        data -= mean

        # data_ = F.pad(data, (pad1, pad2, 0, 0), mode="reflect")
        data_ = F.pad(data, (padding, padding, 0, 0), mode="reflect")
        # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter) ** 0.5))
        std = F.avg_pool2d(torch.abs(data_), kernel_size=(1, filter), stride=(1, stride))
        std = torch.mean(std, dim=(1,), keepdim=True)  ## keep relative amplitude between channels
        std = F.interpolate(std, scale_factor=(1, stride), mode="bilinear", align_corners=False)[:, :, :nx, :nt]
        std[std == 0.0] = 1.0
        data = data / std

        # data = log_transform(data)

    return data


class MergeFrequency(nn.Module):
    """
    Merge frequency dimension to 1 using a linear layer.
    """

    def __init__(self, dim_in):
        super().__init__()
        # self.linear = nn.Sequential(nn.Linear(dim_in, 1), nn.ReLU())
        self.linear = nn.Linear(dim_in, 1)

    def forward(self, x):
        # x: nb, nc, nf, nt
        x = x.permute(0, 1, 3, 2)  # nb, nc, nt, nf
        x = self.linear(x).squeeze(-1)  # nb, nc, nt
        return x


class MergeBranch(nn.Module):
    """
    Merge two branches of the same dimension.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # self.conv = nn.Sequential(nn.Conv2d(dim_in, dim_out, 1), nn.ReLU())
        self.conv = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x1, x2):
        return self.conv(torch.cat((x1, x2), dim=1))


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=128 + 1,
        hop_length=4,
        window_fn=torch.hann_window,
        magnitude=True,
        normalize_freq=False,
        discard_zero_freq=True,
        **kwargs,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.magnitude = magnitude
        self.discard_zero_freq = discard_zero_freq
        self.normalize_freq = normalize_freq
        self.register_buffer("window", window_fn(n_fft))
        self.window_fn = window_fn

    def forward(self, x):
        # window = self.window_fn(self.n_fft).to(x.device)
        """
        x: bt, ch, nt
        """
        nb, nc, nt = x.shape
        x = x.view(-1, nt)  # nb*nc, nt
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            center=True,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        # stft = stft[..., : x.shape[-1] // self.hop_length, :]  # nb*nc*nx, nf, nt, 2
        # stft = stft[..., :-1, :]
        if self.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        nf, nt, _ = stft.shape[-3:]
        if self.magnitude:
            stft = torch.norm(stft, dim=-1, keepdim=False).view(nb, nc, nf, nt)  # nb, nc, nf, nt
        else:
            stft = stft.view(nb, nc, nf, nt, 2)  # nb, nc, nf, nt, 2
            stft = stft.permute(0, 1, 4, 2, 3).view(nb, nc * 2, nf, nt)  # nb, nc*2, nf, nt

        if self.normalize_freq:
            vmax = torch.max(torch.abs(stft), dim=-2, keepdim=True)[0]
            vmax[vmax == 0.0] = 1.0
            stft = stft / vmax

        return stft


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# def Upsample(dim, dim_out = None):
#     return nn.Sequential(
#         nn.Upsample(scale_factor = (1, 2), mode = 'nearest'),
#         nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
#     )

# def Downsample(dim, dim_out = None):
#     return nn.Conv2d(dim, default(dim_out, dim), (1, 4), (1, 2), (0, 1))


def Upsample(dim, dim_out, stride=(1, 4)):

    return nn.ConvTranspose2d(dim, dim_out, stride, stride)


def Downsample(dim, dim_out, stride=(1, 4)):

    return nn.Sequential(
        Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=stride[0], s2=stride[1]),
        nn.Conv2d(dim * stride[0] * stride[1], dim_out, 1),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=(1, 7), padding=(0, 3)):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=padding)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, kernel_size=(1, 7), padding=(0, 3), time_emb_dim=None, classes_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2))
            if exists(time_emb_dim) or exists(classes_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, kernel_size=kernel_size, padding=padding)
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size, padding=padding)
        self.res_conv = nn.Conv2d(dim, dim_out, (1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# model


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 4),
        channels=3,
        attn_dim_head=32,
        attn_heads=4,
        kernel_size=(1, 7),
        scale_factor=None,
        moving_norm=(1024, 128),
        linear_attn=False,
        add_stft=False,
        log_scale=False,
        add_polarity=False,
        add_event=False,
        add_prompt=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in kernel_size)
        self.scale_factor = (
            [tuple(map(lambda k: k // 2 + 1, kernel_size))] * 4 if scale_factor is None else scale_factor
        )
        self.moving_norm = moving_norm
        self.log_scale = log_scale
        self.add_stft = add_stft
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.add_prompt = add_prompt
        self.linear_attn = linear_attn

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=self.kernel_size, padding=self.padding)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, kernel_size=self.kernel_size, padding=self.padding),
                        (
                            ResnetBlock(dim_in, dim_in, kernel_size=self.kernel_size, padding=self.padding)
                            if self.linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.linear_attn else nn.Identity(),
                        (Downsample(dim_in, dim_out, self.scale_factor[ind])),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, kernel_size=self.kernel_size, padding=self.padding)
        self.mid_attn = (
            Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
            if self.linear_attn
            else nn.Identity()
        )
        self.mid_block2 = (
            ResnetBlock(mid_dim, mid_dim, kernel_size=self.kernel_size, padding=self.padding)
            if self.linear_attn
            else nn.Identity()
        )
        self.mid_upsample = Upsample(mid_dim, dims[-2], stride=self.scale_factor[-1])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[:-1])):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_out, kernel_size=self.kernel_size, padding=self.padding),
                        (
                            ResnetBlock(dim_out * 2, dim_out, kernel_size=self.kernel_size, padding=self.padding)
                            if self.linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if self.linear_attn else nn.Identity(),
                        (
                            Upsample(dim_out, dim_in, self.scale_factor[ind])
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 1)
                        ),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        # self.final_res_block = ResnetBlock(init_dim * 2, init_dim)
        # self.final_conv = nn.Conv2d(init_dim, self.out_dim, (1, 1))
        self.final_res_block = ResnetBlock(
            init_dim * 2, self.out_dim, kernel_size=self.kernel_size, padding=self.padding
        )

        ## Polarity
        if self.add_polarity:
            self.polarity_init = nn.Conv2d(1, init_dim, kernel_size=self.kernel_size, padding=self.padding)
            dim_in, dim_out = dims[0], dims[1]
            self.polarity_encoder = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            ResnetBlock(dim_in, dim_in, kernel_size=self.kernel_size, padding=self.padding),
                            (
                                ResnetBlock(dim_in, dim_in, kernel_size=self.kernel_size, padding=self.padding)
                                if self.linear_attn
                                else nn.Identity()
                            ),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.linear_attn else nn.Identity(),
                        ]
                    )
                ]
            )
            self.polarity_final = ResnetBlock(
                dim_in * 2, self.out_dim, kernel_size=self.kernel_size, padding=self.padding
            )

        ## Event
        if self.add_event:
            self.event_feature_level = 3
            dim_in, dim_out = dims[self.event_feature_level], dims[self.event_feature_level + 1]
            self.event_final = nn.Sequential(
                ResnetBlock(dim_in, dim_in, kernel_size=self.kernel_size, padding=self.padding),
                Upsample(dim_in, self.out_dim, stride=self.scale_factor[self.event_feature_level]),
            )

        ## STFT
        if self.add_stft:
            self.stft = STFT(n_fft=64 + 1, hop_length=self.scale_factor[0][-1])
            self.kernel_size_stft = [3, self.kernel_size[1]]
            self.padding_stft = [1, self.padding[1]]
            self.spec_init = nn.Sequential(
                nn.Conv2d(
                    channels,
                    init_dim,
                    kernel_size=self.kernel_size_stft,
                    padding=self.padding_stft,
                ),
            )
            self.spec_down = nn.ModuleList([])
            for ind, (dim_in, dim_out) in enumerate(in_out):
                if ind == 0:
                    continue
                self.spec_down.append(
                    nn.ModuleList(
                        [
                            ResnetBlock(dim_in, dim_in, kernel_size=self.kernel_size_stft, padding=self.padding_stft),
                            (
                                ResnetBlock(
                                    dim_in, dim_in, kernel_size=self.kernel_size_stft, padding=self.padding_stft
                                )
                                if self.linear_attn
                                else nn.Identity()
                            ),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self.linear_attn else nn.Identity(),
                            Downsample(dim_in, dim_out, self.scale_factor[ind]),
                            MergeFrequency(32),
                            MergeBranch(dim_in * 2, dim_in),
                        ]
                    )
                )
            self.mid_stft = nn.Sequential(
                ResnetBlock(mid_dim, mid_dim, kernel_size=self.kernel_size_stft, padding=self.padding_stft),
                MergeFrequency(32),
            )
            self.mid_merge = MergeBranch(mid_dim * 2, mid_dim)

    def forward(self, x):
        # unet

        x = moving_normalize(x, filter=self.moving_norm[0], stride=self.moving_norm[1])
        if self.log_scale:
            x = log_transform(x)

        # origin
        x_origin = x.clone()

        x = self.init_conv(x)
        if self.add_polarity:
            x_polarity = self.polarity_init(x_origin[:, -1:, :, :])
            # x_polarity = self.polarity_init(torch.clamp(x_origin[:, -1:, :, :], -1.0, 1.0))

        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            if self.linear_attn:
                x = block2(x)
                x = attn(x)
                h.append(x)

            x = downsample(x)

        if self.add_stft:
            nb, nc, nx, nt = x_origin.shape
            x_stft = x_origin.permute(0, 2, 1, 3).reshape(nb * nx, nc, nt)  # nb*nx, nc, nt
            x_stft = self.stft(x_stft)  # nb*nx, nc, nf, nt
            # if self.training:
            if True:
                sgram = x_stft.clone()
            else:
                sgram = None
            x_stft = self.spec_init(x_stft)
            step = 2 if self.linear_attn else 1
            for i, (block1, block2, attn, downsample, merge_freq, merge_branch) in enumerate(self.spec_down):
                x_stft = block1(x_stft)  # nb*nx, nc, nf, nt
                x_stft_m = merge_freq(x_stft)  # nb*nx, nc, nt
                x_stft_m = x_stft_m.view(nb, nx, *x_stft_m.shape[-2:]).permute(0, 2, 1, 3)  # nb, nc, nx, nt
                h[(i + 1) * step] = merge_branch(h[(i + 1) * step], x_stft_m)

                if self.linear_attn:
                    x_stft = block2(x_stft)
                    x_stft_m = merge_freq(x_stft)  # nb*nx, nc, nt
                    x_stft_m = x_stft_m.view(nb, nx, *x_stft_m.shape[-2:]).permute(0, 2, 1, 3)  # nb, nc, nx, nt
                    x_stft_m = attn(x_stft_m)
                    h[(i + 1) * step + 1] = merge_branch(h[(i + 1) * step + 1], x_stft_m)

                x_stft = downsample(x_stft)

            x_stft = self.mid_stft(x_stft)
            x_stft = x_stft.view(nb, nx, *x_stft.shape[-2:]).permute(0, 2, 1, 3)  # nb, nc, nx, nt

        x = self.mid_block1(x)
        if self.add_stft:
            x = self.mid_merge(x, x_stft)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        x = self.mid_upsample(x)

        feature_level = 3
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            if self.linear_attn:
                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x)
                x = attn(x)

            if self.add_event and (feature_level == self.event_feature_level):
                x_event = x.clone()

            x = upsample(x)

            feature_level -= 1

        if self.add_polarity:
            x_polarity_aux = x.clone()

        x = torch.cat((x, r), dim=1)
        # x = torch.cat((x, h.pop()), dim=1)

        # x = self.final_res_block(x)
        # out_phase = self.final_conv(x)
        out_phase = self.final_res_block(x)

        # polarity
        if self.add_polarity:
            for block1, block2, attn in self.polarity_encoder:
                x_polarity = block1(x_polarity)
                x_polarity = block2(x_polarity)
                x_polarity = attn(x_polarity)

            x_polarity = torch.cat((x_polarity, x_polarity_aux), dim=1)
            out_polarity = self.polarity_final(x_polarity)
        else:
            out_polarity = None

        # event
        if self.add_event:
            out_event = self.event_final(x_event)
        else:
            out_event = None

        out = {"phase": out_phase, "polarity": out_polarity, "event": out_event}
        # if self.add_stft and self.training:
        if self.add_stft:
            out["spectrogram"] = sgram.squeeze(2)  ## nb, nc, nx, nf, nt -> nb, nc, nf, nt
        return out


# example

if __name__ == "__main__":

    from torchinfo import summary
    from torchviz import make_dot

    model = UNet(
        dim=8,
        out_dim=16,
        dim_mults=(1, 2, 4, 8),
        add_polarity=True,
        add_event=True,
        add_stft=True,
        linear_attn=False,
    )

    # print(model)
    data = torch.randn(7, 3, 1, 4096)

    summary(model, input_size=data.shape, depth=5)

    model.to("cpu")
    out = model(data)

    dot = make_dot(out["phase"], params=dict(model.named_parameters()))
    dot.render("unet", format="png")

    # for k, v in out.items():
    #     if v is not None:
    #         print(f"{k}: {v.shape}")

    # model = UNet(
    #     dim=16,
    #     out_dim=32,
    #     dim_mults=(1, 2, 4, 8),
    #     add_polarity=True,
    #     add_event=True,
    #     add_stft=False,
    # ).cuda()

    # # print(model)
    # data = torch.randn(7, 3, 1, 4096).cuda()

    # summary(model, input_size=data.shape, depth=5)

    # out = model(data)

    # for k, v in out.items():
    #     if v is not None:
    #         print(f"{k}: {v.shape}")

    # model = Unet(
    #     dim = 8,
    #     out_dim=16,
    #     dim_mults = (1, 2, 4, 8),
    #     linear_attn=True,
    # ).cuda()

    # summary(model, input_size=(7, 3, 1, 128), depth=5)
