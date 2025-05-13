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


class TransformerClassifier(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_layers=4,
                 num_heads=12,
                 hidden_dim=3072,
                 dropout=0.1):
        super().__init__()

        self.input_norm = nn.LayerNorm(768)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_norm(x)
        encoded = self.transformer(x)
        pooled = encoded.squeeze(1)
        logits = self.classifier(pooled)
        return logits


class VideoDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionTransformer(
            input_resolution=224, patch_size=14,
            width=1024, layers=24, heads=16, output_dim=768,
            kernel_size=1, num_frames=8,
            drop_path=0.1, checkpoint_num=24,
            dropout=0.,
        )
        self.classifier = TransformerClassifier()

    def forward(self, x):
        x = self.classifier(self.encoder(x))
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = VideoDetection()
    model.encoder.load_state_dict(torch.load('vision_encoder.pth'))
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.to(device)
    input_tensor = torch.randn(size=[1, 3, 8, 224, 224]).to(device)
    print(input_tensor.size())
    output_tensor = model(input_tensor)
    print(output_tensor.size())
