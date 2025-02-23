import torch
import torch.nn as nn
from maniskill2_learn.utils.torch import ExtendedModuleList
from ..builder import BACKBONES, build_backbone
from ..modules import build_attention_layer

from maniskill2_learn.utils.torch import ExtendedModule
from maniskill2_learn.utils.data import split_dim, GDict
from maniskill2_learn.networks.modules.cnn_modules.model_getter import get_resnet
from maniskill2_learn.networks.modules.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)

from pytorch3d.transforms import quaternion_to_matrix

from .mlp import LinearMLP

import numpy as np
import open3d as o3d
import time


class TransformerBlock(ExtendedModule):
    def __init__(self, attention_cfg, mlp_cfg, dropout=None):
        super(TransformerBlock, self).__init__()
        self.attn = build_attention_layer(attention_cfg)
        self.mlp = build_backbone(mlp_cfg)
        assert mlp_cfg.mlp_spec[0] == mlp_cfg.mlp_spec[-1] == attention_cfg.embed_dim

        self.ln1 = nn.LayerNorm(attention_cfg.embed_dim)
        self.ln2 = nn.LayerNorm(attention_cfg.embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x, mask=None, history=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim], the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length], a mask for disallowing attention to padding tokens
        :param history: [B, H, E] [history length, batch size, embed_dim], the histroy embeddings
        :param ret_history, if we return the emebeding in previous segments
        :param histroy_len: int, the maximum number of history information we store
        :return: [B, N, E] [batch size, length, length] a single tensor containing the output from the Transformer block
        """
        ret_history = x if history is None else torch.cat([history, x], dim=1)
        o = self.attn(x, mask, history)
        x = x + o
        x = self.ln1(x)
        o = self.mlp(x)
        o = self.dropout(o)
        x = x + o
        x = self.ln2(x)
        return x, ret_history.detach()


class AddPositionalEncoding(ExtendedModule):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1, max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x):
        """
        Args:
        x: A tensor of shape [batch size, length, d_model]
        """
        x = self.input_dropout(x)
        timing = self.timing_table[None, : x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing


@BACKBONES.register_module()
class TransformerEncDec(ExtendedModule):
    def __init__(
        self,
        block_cfg,
        pooling_cfg=None,
        mlp_cfg=None,
        num_blocks=6,
        with_task_embedding=False,
    ):
        super(TransformerEncDec, self).__init__()

        if with_task_embedding:
            embed_dim = block_cfg["attention_cfg"]["embed_dim"]
            self.task_embedding = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.xavier_normal_(self.task_embedding)
        self.with_task_embedding = with_task_embedding

        self.attn_blocks = ExtendedModuleList(
            [TransformerBlock(**block_cfg) for i in range(num_blocks)]
        )
        self.pooling = (
            None
            if pooling_cfg is None
            else build_attention_layer(
                pooling_cfg, default_args=dict(type="AttentionPooling")
            )
        )
        self.global_mlp = build_backbone(mlp_cfg) if mlp_cfg is not None else None

    def forward(self, x, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] or [B, N, F] A single tensor or a series of tensor containing the output from the Transformer
        """
        assert x.ndim == 3
        B, N, E = x.shape
        assert mask is None or list(mask.shape) == [
            B,
            N,
            N,
        ], f"{mask.shape} {[B, N, N]}"
        if mask is None:
            mask = torch.ones(B, N, N, dtype=x.dtype, device=x.device)
        mask = mask.type_as(x)
        if self.with_task_embedding:
            one = torch.ones_like(mask[:, :, 0])
            mask = torch.cat([one.unsqueeze(1), mask], dim=1)  # (B, N+1, N)
            one = torch.ones_like(mask[:, :, 0])
            mask = torch.cat([one.unsqueeze(2), mask], dim=2)  # (B, N+1, N+1)
            x = torch.cat(
                [torch.repeat_interleave(self.task_embedding, x.size(0), dim=0), x],
                dim=1,
            )

        for i, attn in enumerate(self.attn_blocks):
            x = attn(x, mask)[0]
        if self.pooling is not None:
            x = self.pooling(x)
        if self.global_mlp is not None:
            x = self.global_mlp(x)
        return x

    def get_atten_score(self, x, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] or [B, N, F] A single tensor or a series of tensor containing the output from the Transformer
        """
        assert x.ndim == 3
        B, N, E = x.shape
        assert mask is None or mask.shape == [B, N, N], f"{mask.shape} {[B, N, N]}"
        if mask is None:
            mask = torch.ones(B, N, N, dtype=x.dtype, device=x.device)
        mask = mask.type_as(x)

        ret = []
        for attn in self.attn_blocks:
            score = attn.attn.get_atten_score(x)
            x = attn(x, mask)[0]
            ret.append(score)
        return torch.stack(ret, dim=0)


@BACKBONES.register_module()
class TransformerEncDecEmbedding(TransformerEncDec, MultiImageObsEncoder):
    def __init__(
        self,
        shape_meta,
        block_cfg,
        pooling_cfg=None,
        mlp_cfg=None,
        num_blocks=6,
        with_task_embedding=False,
        rgb_model=get_resnet("resnet18"),
        pcd_model=None,
        resize_shape=None,
        crop_shape=None,  # [76,76],
        random_crop=True,
        # replace BatchNorm with GroupNorm
        use_group_norm=False,
        # use single rgb model for all rgb inputs
        share_rgb_model=True,
        use_pcd_model=False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm=False,
        n_obs_steps=1,
        embedding_size=256,
    ):
        TransformerEncDec.__init__(
            self,
            block_cfg=block_cfg,
            pooling_cfg=pooling_cfg,
            mlp_cfg=mlp_cfg,
            num_blocks=num_blocks,
            with_task_embedding=with_task_embedding,
        )
        MultiImageObsEncoder.__init__(
            self,
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            pcd_model=pcd_model,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            use_pcd_model=use_pcd_model,
            imagenet_norm=imagenet_norm,
            n_obs_steps=n_obs_steps,
        )

        self.embedding_size = embedding_size

        self.embedding_model_map = dict()
        for key in self.low_dim_keys:
            self.embedding_model_map[key] = nn.Sequential(
                nn.Mish(),
                nn.Linear(self.key_shape_map[key], embedding_size),
                nn.Mish(),
                nn.Linear(embedding_size, embedding_size),
            )
        rgb_keys = self.rgb_keys
        if self.share_rgb_model:
            rgb_keys = ["rgb"]
        for key in rgb_keys:
            self.embedding_model_map[key] = nn.Sequential(
                nn.Mish(),
                nn.Linear(self.feature_shape_map[key], embedding_size),
                nn.Mish(),
                nn.Linear(embedding_size, embedding_size),
            )

    def forward(self, input_dict, mask=None):
        """
        :param input_dict: Dict
        We should process the input data x to the TransformerEncDec's forward
        """
        batch_size = None
        tokens = list()
        horizon = self.n_obs_steps
        # Preprocess img model
        input_dict = self.preprocess(input_dict)
        # process rgb input
        for key in self.rgb_keys:
            img = input_dict[key]
            if isinstance(img, list):
                img = img[0]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            if len(img.shape) == 5:  # (B,L,C,H,W)
                assert (
                    img.shape[1] == horizon
                ), "The input horizon {} is not the same as expected obs length {}!".format(
                    img.shape[1], self.n_obs_steps
                )
                img = img.reshape(batch_size * horizon, *img.shape[2:])  # (B*L,C,H,W)
            assert (
                img.shape[1:] == self.key_shape_map[key]
            ), f"{img.shape[1:]} != {self.key_shape_map[key]}"  # (C,H,W)
            img = self.key_transform_map[key](img)
            feature = self.key_model_map[key](img)  # (B*L,D)
            token = self.embedding_model_map[key](feature)  # (B*L,E)
            token = token.reshape(batch_size, horizon, token.shape[-1])  # (B,L,E)
            tokens.append(token)

        for key in self.low_dim_keys:
            token = self.embedding_model_map[key](input_dict[key])
            tokens.append(token)  # (B,L,E)

        x = torch.cat(tokens, dim=1)  # B, L, E
        pred_seq = TransformerEncDec.forward(self, x, mask)

        return pred_seq
