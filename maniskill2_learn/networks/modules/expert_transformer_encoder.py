from typing import Dict, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from maniskill2_learn.utils.diffusion.torch import dict_apply, replace_submodules
from maniskill2_learn.networks.modules.cnn_modules.model_getter import get_resnet
from maniskill2_learn.networks.modules.multi_image_obs_encoder import *
from maniskill2_learn.networks.backbones.rl_cnn import CNNBase
from maniskill2_learn.networks.backbones.pointnet import PointNet
from maniskill2_learn.utils.torch import no_grad
from maniskill2_learn.networks.builder import MODELNETWORKS, build_model
from maniskill2_learn.networks.modules.block_utils import SimpleMLP as MLP
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            y (torch.Tensor): image features
                shape (B, T_img, D_img) where n is the dim of the latents
        """
        _, T_img, _ = y.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        k, v = self.to_kv(y).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


@MODELNETWORKS.register_module()
class MultiImageObsEncoderWithDemo(MultiImageObsEncoder):
    """
    new keys: demo_rgb, demo_depth, demo_state, demo_action, action
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]] = get_resnet("resnet18"),
        pcd_model: dict = None,
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[
            Tuple[int, int], Dict[str, tuple], None
        ] = None,  # [104,104], # [76,76],
        random_crop: bool = True,
        random_rotation: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = True,
        use_pcd_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        n_obs_steps: int = 1,
        max_demo_steps: int = 1,
        token_dim=256,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__(
            shape_meta,
            rgb_model,
            pcd_model,
            resize_shape,
            crop_shape,
            random_crop,
            random_rotation,
            use_group_norm,
            share_rgb_model,
            use_pcd_model,
            imagenet_norm,
            n_obs_steps,
            get_output_dim=False,
        )
        self.action_keys = []
        self.max_demo_steps = max_demo_steps
        self.token_dim = token_dim

        obs_shape_meta = shape_meta["obs"]
        act_shape_meta = shape_meta["actions"]

        # init key_model_map & key_transform_map
        for key, attr in act_shape_meta.items():
            shape = attr["shape"]
            act_type = attr["type"]
            if "action" in act_type:
                self.action_keys.append(key)
                self.key_model_map[key] = MLP(
                    input_dim=np.prod(shape), output_dim=token_dim
                )
                if type(shape) == int:
                    shape = (shape,)
                self.key_shape_map[key] = shape
            else:
                raise NotImplementedError

        self.frame_pos_emb = nn.Parameter(torch.randn(1, max_demo_steps, token_dim))
        self.action_pos_emb = nn.Parameter(torch.randn(1, max_demo_steps, token_dim))

        self.obs_fuse = MLP(
            input_dim=np.prod(obs_shape_meta["state"]["shape"]) + 512 * 2,
            output_dim=token_dim,
        )  # 512 is the feature size of images
        self.demo_fuse = MLP(
            input_dim=np.prod(obs_shape_meta["state"]["shape"]) + 512 * 2,
            output_dim=token_dim,
        )  # 512 is the feature size of images

        self.cross_att = MaskedCrossAttention(token_dim, token_dim)
        self.global_1d_max_pool = nn.AdaptiveMaxPool1d(1)

        self.out_feature_dim = self.output_shape()

    def forward(self, obs_dict, act_dict):
        batch_size = None
        demo_batch = None
        features = list()
        demo_feats = list()
        horizon = self.n_obs_steps
        # Preprocess img model
        # !!!!!!!
        obs_dict = self.preprocess(obs_dict)
        # process rgb input
        if len(self.rgb_keys):
            if self.share_rgb_model:
                # pass all rgb obs to rgb model
                imgs = list()
                demo_imgs = list()
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    if "demo" in key:
                        n_demo_steps = img.shape[1]
                    #     assert n_demo_steps == self.max_demo_steps, "should be equivalent to the max demo steps!"
                    if batch_size is None and ("demo" not in key):
                        batch_size = img.shape[0]
                    else:
                        if "demo" not in key:
                            assert batch_size == img.shape[0]

                    if demo_batch is None and ("demo" in key):
                        demo_batch = img.shape[0]
                    else:
                        if "demo" in key:
                            assert demo_batch == img.shape[0]

                    if len(img.shape) == 5:  # (B,L,C,H,W)
                        if "demo" not in key:
                            assert (
                                img.shape[1] == horizon
                            ), "The input horizon {} is not the same as expected obs length {}!".format(
                                img.shape[1], self.n_obs_steps
                            )
                        img = img.reshape(
                            img.shape[0] * img.shape[1], *img.shape[2:]
                        )  # (B*L,C,H,W)
                    if "demo" not in key:
                        assert (
                            img.shape[1:] == self.key_shape_map[key]
                        ), f"{img.shape[1:]} != {self.key_shape_map[key]}"  # (C,H,W)
                    img = self.key_transform_map[key](img)
                    if "demo" in key:
                        demo_imgs.append(img)
                    else:
                        imgs.append(img)
                # (N*B*L,C,H,W)
                imgs = torch.cat(imgs, dim=0)
                demo_imgs = torch.cat(demo_imgs, dim=0)
                # (N*B*L,D)
                feature = self.key_model_map["rgb"](imgs)
                demo_feat = self.key_model_map["rgb"](demo_imgs)
                if "rgb" in self.feature_shape_map:
                    self.feature_shape_map["rgb"] = feature.shape[-1]
                # (N,B,L,D)
                feature = feature.reshape(-1, batch_size, horizon, *feature.shape[1:])
                # Demo: (N,B,L,D)
                demo_feat = demo_feat.reshape(
                    -1, demo_batch, n_demo_steps, *demo_feat.shape[1:]
                )

                feature = torch.moveaxis(feature, 0, 1)  # (B,N,L,D)
                feature = torch.moveaxis(feature, 1, 2)  # (B,L,N,D)
                demo_feat = torch.moveaxis(demo_feat, 0, 1)  # Demo: (B,N,L,D)
                demo_feat = torch.moveaxis(demo_feat, 1, 2)  # Demo: (B,L,N,D)

                # (B,L,N*D)
                feature = feature.reshape(batch_size, horizon, -1)
                demo_feat = demo_feat.reshape(demo_batch, n_demo_steps, -1)  # (B,L,N*D)
                features.append(feature)
                demo_feats.append(demo_feat)
            else:
                raise NotImplementedError
                ## This part of code has not been varified
                # run each rgb obs to independent models
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    if isinstance(img, list):
                        img = img[0]
                    if batch_size is None:
                        batch_size = img.shape[0]
                    else:
                        assert batch_size == img.shape[0]
                    if len(img.shape) == 5:  # (bs, length, channel, h, w)
                        img = img.reshape(batch_size * img.shape[1], -1)
                    assert img.shape[2:] == self.key_shape_map[key]  # bs, horizon
                    img = self.key_transform_map[key](img)
                    feature = self.key_model_map[key](img)
                    if "rgb" in self.feature_shape_map:
                        self.feature_shape_map[key] = feature.shape[-1]
                    if "demo" in key:
                        demo_feats.append(feature)
                    else:
                        features.append(feature)

        ## process pcd input
        # if self.use_pcd_model and len(self.pcd_keys):
        #     pcd_obs_dict = {k: v for k, v in obs_dict.items() if k in self.pcd_keys}
        #     for key in pcd_obs_dict:
        #         if batch_size is None:
        #             batch_size = pcd_obs_dict[key].shape[0]
        #         pcd_obs_dict[key] = pcd_obs_dict[key].reshape(batch_size * horizon, *pcd_obs_dict[key].shape[2:])
        #     feature = self.key_model_map["pcd"](pcd_obs_dict)
        #     feature = feature.reshape(batch_size, -1)
        #     features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None and ("demo" not in key):
                batch_size = data.shape[0]
            else:
                if "demo" not in key:
                    assert batch_size == data.shape[0], f"{batch_size}, {data.shape}"
            assert (
                data.shape[2:] == self.key_shape_map[key]
            ), f"{key}, {data.shape}, {self.key_shape_map[key]}"  # bs, horizon
            seq_len = data.shape[1]
            if "demo" in key:
                data = data.view(demo_batch, seq_len, -1)  # (B,L,D)
                demo_feats.append(data)
            else:
                data = data.reshape(batch_size, seq_len, -1)
                features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)  # (B,L,feature_dim)
        demo_res = torch.cat(demo_feats, dim=-1)  # (B,L,feature_dim)

        hist_obs_tokens = self.obs_fuse(result)
        hist_obs_tokens += self.frame_pos_emb[:, : hist_obs_tokens.shape[1]]
        demo_obs_tokens = self.demo_fuse(demo_res)  # bs, demo_len, token_dim
        demo_obs_tokens += self.frame_pos_emb[:, : demo_obs_tokens.shape[1]]

        act_feat = None
        demo_act_tokens = None
        if len(self.action_keys):
            for key in self.action_keys:
                data = act_dict[key]
                if "demo" in key:
                    assert demo_batch == data.shape[0]
                else:
                    assert batch_size == data.shape[0], f"{batch_size}, {data.shape}"
                assert (
                    data.shape[2:] == self.key_shape_map[key]
                ), f"{key}, {data.shape}, {self.key_shape_map[key]}"  # bs, horizon

                seq_len = data.shape[1]
                # data = data.reshape(batch_size, -1)
                data = self.key_model_map[key](data)
                if "demo" in key:
                    demo_act_tokens = data
                    demo_act_tokens = demo_act_tokens.view(demo_batch, seq_len, -1)
                    demo_act_tokens += self.action_pos_emb[
                        :, : demo_act_tokens.shape[1]
                    ]
                else:
                    hist_act_tokens = data
                    hist_act_tokens = hist_act_tokens.view(batch_size, seq_len, -1)
                    hist_act_tokens += self.action_pos_emb[
                        :, : hist_act_tokens.shape[1]
                    ]

        assert (hist_act_tokens is not None) and (demo_act_tokens is not None)

        T = demo_obs_tokens.shape[1]
        demo_tokens = torch.zeros(
            [demo_obs_tokens.shape[0], T * 2, self.token_dim],
            dtype=torch.float32,
            device=self.device,
        )
        demo_tokens[:, : T * 2 : 2, :] = demo_obs_tokens
        demo_tokens[:, 1 : T * 2 : 2, :] = demo_act_tokens

        T = hist_obs_tokens.shape[1]
        hist_tokens = torch.zeros(
            [hist_obs_tokens.shape[0], T * 2, self.token_dim],
            dtype=torch.float32,
            device=self.device,
        )
        hist_tokens[:, : T * 2 : 2, :] = hist_obs_tokens
        hist_tokens[:, 1 : T * 2 - 1 : 2, :] = hist_act_tokens[
            :, : T - 1
        ]  # Remove the last one (ground truth)

        res = self.cross_att(hist_tokens, demo_tokens)
        res = res.reshape(res.shape[0], -1)  # max pooling over the sequence length dim
        # res = self.global_1d_max_pool(res.permute(0, 2, 1)).squeeze(-1)  # max pooling over the sequence length dim

        return res  # B, feature_size (feature_size = resenet_fea+low_dim)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        example_act_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        act_shape_meta = self.shape_meta["actions"]
        batch_size = 1
        horizon = self.n_obs_steps
        for key, attr in obs_shape_meta.items():
            shape = self.key_shape_map[key]
            this_obs = torch.zeros(
                (batch_size, horizon) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        for key, attr in act_shape_meta.items():
            shape = self.key_shape_map[key]
            this_horizon = horizon
            if "demo" not in key:
                this_horizon = horizon - 1
            this_act = torch.zeros(
                (batch_size, this_horizon) + shape, dtype=self.dtype, device=self.device
            )
            example_act_dict[key] = this_act
        example_output = self.forward(example_obs_dict, example_act_dict)
        output_shape = list(example_output.shape[1:])
        if len(output_shape) == 1:
            output_shape = output_shape[0]
        return output_shape
