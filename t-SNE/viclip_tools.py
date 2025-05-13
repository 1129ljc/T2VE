import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from collections import OrderedDict
from timm.models.layers import DropPath


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, dropout=0.):
        super().__init__()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("drop2", nn.Dropout(dropout)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, drop_path=dpr[idx], dropout=dropout))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, input_resolution, patch_size, width, layers, heads, output_dim=None,
            kernel_size=1, num_frames=8, drop_path=0., checkpoint_num=0, dropout=0.,
            temp_embed=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3, width,
            (kernel_size, patch_size, patch_size),
            (kernel_size, patch_size, patch_size),
            (0, 0, 0), bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout)

        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None

        self.dropout = nn.Dropout(dropout)

    def mask_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape

        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1

        masked_indices = masked_indices.bool()

        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])

    def forward(self, x, masking_prob=0.0):
        x = self.conv1(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)

        x = self.ln_post(x)

        if self.proj is not None:
            x = self.dropout(x[0]) @ self.proj
        else:
            x = x.permute(1, 0, 2)

        return x


def normalize(data):
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    return (data / 255.0 - v_mean) / v_std


def init_viclip_model(device):
    video_encoder = VisionTransformer(
        input_resolution=224, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=1, num_frames=8,
        drop_path=0.1, checkpoint_num=24,
        dropout=0.,
    ).to(device)
    video_encoder.load_state_dict(torch.load('vision_encoder.pth'))
    return video_encoder


def get_video_feats(video_path, model, device):
    frames = []
    pic_names = os.listdir(video_path)
    for pic_name in pic_names:
        pic_path = os.path.join(video_path, pic_name)
        frame = cv2.imread(pic_path)
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    step = len(frames) // 8
    vid_list = frames[::step][:8]
    vid_list = [cv2.resize(x[:, :, ::-1], (224, 224)) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    image = vid_tube.permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad():
        clip_feat = model(image)
        clip_feat = clip_feat.float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
        clip_feat = torch.squeeze(clip_feat).cpu().numpy()
    return clip_feat


def get_videos_viclip_feats(video_paths, model, device):
    video_feats = []
    for index in tqdm(range(len(video_paths))):
        video_feats.append(get_video_feats(video_paths[index], model, device))
    video_feats = np.vstack(video_feats)
    return video_feats
