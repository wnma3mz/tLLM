# coding: utf-8
# Modify from https://github.com/deepseek-ai/Janus/blob/main/janus/models/vq_model.py
from dataclasses import dataclass, field
from typing import List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


class ConvBlock(nn.Module):
    def __init__(self, res_block, attn_block):
        super().__init__()
        self.res = res_block
        self.attn = attn_block


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        z_channels=256,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = []
        for i_level in range(self.num_resolutions):
            # res & attn
            res_block = []
            attn_block = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block = ConvBlock(res_block, attn_block)
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = []
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = []
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = []
        for i_level in reversed(range(self.num_resolutions)):
            # res & attn
            res_block = []
            attn_block = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block = ConvBlock(res_block, attn_block)
            # conv_block.res = res_block
            # conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def __call__(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def normalize(x: mx.array, p: int, axis: int):
    norm = mx.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    # Normalize the input array
    normalized_array = x / norm
    return normalized_array


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if self.l2_norm:
            self.embedding.weight = normalize(self.embedding.weight, p=2, axis=-1)

    def __call__(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z = mx.einsum("b c h w -> b h w c", z)
        # z_flattened = z.view(-1, self.e_dim)
        z_flattened = z.reshape(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            # z = F.normalize(z, p=2, dim=-1)
            # z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            # embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
            z = normalize(z, p=2, axis=-1)
            z_flattened = normalize(z_flattened, p=2, axis=-1)
            embedding = normalize(self.embedding, p=2, axis=-1)
        else:
            embedding = self.embedding

        d = (
            mx.sum(z_flattened**2, axis=1, keepdim=True)
            + mx.sum(embedding**2, axis=1)
            - 2 * mx.einsum("bd,dn->bn", z_flattened, mx.einsum("n d -> d n", embedding))
        )

        min_encoding_indices = mx.argmin(d, axis=1)
        # z_q = embedding[min_encoding_indices].view(z.shape)
        z_q = embedding[min_encoding_indices].reshape(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None

        # compute loss for embedding
        # if self.training:
        #     vq_loss = torch.mean((z_q - z.detach()) ** 2)
        #     commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
        #     entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z)  # .detach()

        # reshape back to match original input shape
        z_q = mx.einsum("b h w c -> b c h w", z_q)

        return (
            z_q,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        )

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            # embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
            embedding = normalize(self.embedding.weight, p=2, axis=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.transpose(0, 3, 1, 2)  # .contiguous()
            else:
                # z_q = z_q.view(shape)
                z_q = z_q.reshape(shape)
        return z_q


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # in torch
        # b, c, h, w = q.shape
        # q = q.reshape(b, c, h * w)
        # q = q.transpose(0, 2, 1)  # b,hw,c
        # k = k.reshape(b, c, h * w)  # b,c,hw
        # w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        b, h, w, c = q.shape  # in mlx
        q = q.reshape(b, h * w, c)
        k = k.reshape(b, h * w, c)  # b,hw,c
        k = k.transpose(0, 2, 1)  # b,c,hw

        w_ = mx.matmul(q, k)

        w_ = w_ * (int(c) ** (-0.5))
        # w_ = F.softmax(w_, dim=2)
        w_ = mx.softmax(w_, axis=2)

        # attend to values
        # v = v.reshape(b, c, h * w)
        v = v.reshape(b, h * w, c)
        v = v.transpose(0, 2, 1)
        w_ = w_.transpose(0, 2, 1)
        # h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = mx.matmul(v, w_)
        # h_ = h_.reshape(b, c, h, w)
        h_ = h_.transpose(0, 2, 1)
        h_ = h_.reshape(b, h, w, c)

        h_ = self.proj_out(h_)
        return x + h_


def nonlinearity(x):
    # swish
    # return x * torch.sigmoid(x)
    return x * mx.sigmoid(x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        # return nn.GroupNorm(
        #     num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        # )
        return nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-6, affine=True, pytorch_compatible=True)
    elif norm_type == "batch":
        raise NotImplementedError("SyncBatchNorm not supported")
        # return nn.SyncBatchNorm(in_channels)


# from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        x = upsample_nearest(x, scale=2)
        # if x.dtype != torch.float32:
        #     x = F.interpolate(x.to(torch.float), scale_factor=2.0, mode="nearest").to(
        #         torch.bfloat16
        #     )
        # else:
        #     x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = mx.pad(x, pad, mode="constant", value=0)
            # x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            raise NotImplementedError("F.avg_pool2d not supported")
            # x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # self.encoder = Encoder(
        #     ch_mult=config.encoder_ch_mult,
        #     z_channels=config.z_channels,
        #     dropout=config.dropout_p,
        # )
        self.decoder = Decoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )

        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        # self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        # [out_ch, in_ch, h, w] -> [out_ch, h, w, in_ch]
        quant_b = mx.transpose(quant_b, (0, 2, 3, 1))
        # hidden_states = mx.transpose(hidden_states, (0, 2, 3, 4, 1))
        dec = self.decode(quant_b)
        return dec

    def __call__(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
# def VQ_16(**kwargs):
#     return VQModel(
#         ModelArgs(
#             encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs
#         )
#     )


# VQ_models = {"VQ-16": VQ_16}


class vision_head(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = nn.Linear(params["n_embed"], params["image_token_embed"])
        self.vision_activation = nn.GELU()
        self.vision_head = nn.Linear(params["image_token_embed"], params["image_token_size"])

    def __call__(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def sanitize(weights):
    sanitized_weights = {}
    for k, v in weights.items():
        if ".codebook_used" in k:
            continue
        k = k.replace("gen_vision_model.", "")

        if "weight" in k and len(v.shape) == 4:
            # [out_ch, in_ch, h, w] -> [out_ch, h, w, in_ch]
            v = v.transpose(0, 2, 3, 1)

        sanitized_weights[k] = v
    return sanitized_weights


if __name__ == "__main__":
    fname = ...
    state_dict = mx.load(fname)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("gen_vision_model."):
            new_state_dict[k] = v
    state_dict = new_state_dict
    vq_model = VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4]))
    vq_model.load_weights(list(sanitize(state_dict).items()))

    image_token_num_per_image = 576
    generated_tokens = [[10] * image_token_num_per_image]
    parallel_size = 1
    img_size = 384
    patch_size = 16
    dec = vq_model.decode_code(
        generated_tokens, shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size], channel_first=True
    )
    dec = dec.astype(mx.float32)
    # bsz, h, w, channel
    mx_dec = mx.clip((dec + 1) / 2 * 255, 0, 255)
    from PIL import Image
    import numpy as np

    Image.fromarray(np.array(mx_dec[0], dtype=np.uint8)).save("tt.png")
