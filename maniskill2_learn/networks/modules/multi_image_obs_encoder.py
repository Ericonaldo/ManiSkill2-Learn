from typing import Dict, Tuple, Union
import copy
import numpy as np

import torch
import torch.nn as nn
import torchvision

from maniskill2_learn.networks.modules.cnn_modules.crop_randomizer import CropRandomizer
from maniskill2_learn.utils.diffusion.torch import dict_apply, replace_submodules
from maniskill2_learn.networks.modules.cnn_modules.model_getter import get_resnet
from maniskill2_learn.networks.backbones.rl_cnn import CNNBase
from maniskill2_learn.utils.torch import no_grad
from maniskill2_learn.networks.builder import MODELNETWORKS, build_model
from maniskill2_learn.networks.modules.block_utils import SimpleMLP as MLP


@MODELNETWORKS.register_module()
class MultiImageObsEncoder(CNNBase):
    @no_grad
    def preprocess(self, inputs):
        # assert inputs are channel-first; output is channel-first
        if isinstance(inputs, dict):
            if "rgb" in inputs:
                # inputs images must not have been normalized before
                inputs["rgb"] /= 255.0
                if "depth" in inputs:
                    feature = [inputs["rgb"]]
                    depth = inputs["depth"]
                    if isinstance(depth, torch.Tensor):
                        feature.append(depth.float())
                    elif isinstance(depth, np.ndarray):
                        feature.append(depth.astype(np.float32))
                    else:
                        raise NotImplementedError()
                    inputs["rgbd"] = torch.cat(feature, dim=1)
                    inputs.pop("rgb")
                    inputs.pop("depth")
            else:
                for key in inputs:
                    if "rgbd" in key:
                        if len(inputs[key].shape) == 4:  # (B,C,H,W)
                            inputs[key][:, :3, :, :] /= 255.0
                        elif len(inputs[key].shape) == 5:  # (B,L,C,H,W)
                            inputs[key][:, :, :3, :, :] /= 255.0
                        elif len(inputs[key].shape) == 3:  # (C,H,W)
                            inputs[key][:3, :, :] /= 255.0
                        if len(inputs[key].shape) == 4:  # B, 1, C, W, H
                            inputs[key] = inputs[key].unsqueeze(1)

            if "state" in inputs:
                if len(inputs["state"].shape) == 2:  # B, 1, D
                    inputs["state"] = inputs["state"].unsqueeze(1)
            # The following codes have been removed to replay buffer
            # We remove velocity from the state
            #     if isinstance(inputs["state"], torch.Tensor):
            #         inputs["state"] = torch.cat([inputs["state"][...,:9], inputs["state"][...,18:]], axis=-1)
            #     elif isinstance(inputs["state"], np.ndarray):
            #         inputs["state"] = np.concatenate([inputs["state"][...,:9], inputs["state"][...,18:]], axis=-1)
            #     else:
            #         raise NotImplementedError()

        return inputs

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]] = None,
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
        get_output_dim=True,
        output_mlp=False,
        output_vae=False,
        output_dim=128,
        output_hidden_dims=[512, 512],
        use_ep_first_obs=False,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        self.feature_shape_map = dict()
        self.n_obs_steps = n_obs_steps
        self.use_pcd_model = use_pcd_model
        self.use_ep_first_obs = use_ep_first_obs

        if rgb_model is None:
            rgb_model = get_resnet("resnet18")

        obs_shape_meta = shape_meta["obs"]
        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        if use_pcd_model and (pcd_model is not None):
            key_model_map["pcd"] = pcd_model

        self.pcd_keys = list()

        # init key_model_map & key_transform_map
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            if isinstance(shape, list):
                shape = tuple(shape)
            if isinstance(shape, int):
                shape = (shape,)
            key_shape_map[key] = shape
            obs_type = attr["type"]
            if (obs_type == "rgb") or (obs_type == "rgbd"):
                key_model_map[key] = key_model_map["rgb"]
                channel = attr.get("channel", 3)
                shape = tuple([channel, *shape])
                key_shape_map[key] = shape
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                if obs_type == "rgbd":
                    key_model_map[key].conv1 = torch.nn.Conv2d(
                        4,
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False,
                    )

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))
                # configure rotation
                this_rotationer = nn.Identity()
                if random_rotation:
                    this_rotationer = torchvision.transforms.RandomRotation(degrees=180)

                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_rotationer, this_normalizer
                )
                key_transform_map[key] = this_transform
            elif obs_type == "pcd":
                self.pcd_keys.append(key)
            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.output_mlp = None
        self.output_vae = None

        if get_output_dim or output_mlp or output_vae:
            self.out_feature_dim = self.output_shape()
        self.img_feature_dim = self.output_shape(img_fea_only=True)
        if isinstance(self.img_feature_dim, list):
            self.img_feature_dim = self.img_feature_dim[1]

        if output_mlp:
            self.output_mlp = MLP(
                input_dim=self.out_feature_dim,
                output_dim=output_dim,
                hidden_dims=output_hidden_dims,
            )
            self.out_feature_dim = output_dim

            assert (
                self.output_shape() == self.out_feature_dim
            ), "output shape not the same as expected, {}, {}".format(
                self.output_shape(), self.out_feature_dim
            )

        elif output_vae:
            self.output_vae_mu = MLP(
                input_dim=self.out_feature_dim,
                output_dim=output_dim,
                hidden_dims=output_hidden_dims,
            )
            self.output_vae_sigma = MLP(
                input_dim=self.out_feature_dim,
                output_dim=output_dim,
                hidden_dims=output_hidden_dims,
            )
            self.output_vae = [self.output_vae_mu, self.output_vae_sigma]

    def forward(self, obs_dict, ep_first_obs_dict=None, img_fea_only=False):
        batch_size = None
        features = list()
        horizon = self.n_obs_steps
        # Preprocess img model
        # !!!!!!!
        obs_dict = self.preprocess(obs_dict)
        if ep_first_obs_dict is not None:
            ep_first_obs_dict = self.preprocess(ep_first_obs_dict)
        # process rgb input
        if len(self.rgb_keys):
            if self.share_rgb_model:
                # pass all rgb obs to rgb model
                imgs = list()
                for key in self.rgb_keys:
                    # Process ep_first_obs
                    if self.use_ep_first_obs and ep_first_obs_dict is not None:
                        img = ep_first_obs_dict[key]
                        if batch_size is None:
                            batch_size = img.shape[0]
                        img = img.reshape(batch_size, *img.shape[2:])  # (B,C,H,W)
                        assert (
                            len(img.shape) == 4
                        ), f"Ep first image shape mismatch! {img.shape}, {len(img.shape)}"  # (B,C,H,W)
                        assert (
                            img.shape[1:] == self.key_shape_map[key]
                        ), f"{img.shape[1:]} != {self.key_shape_map[key]}"  # (C,H,W)
                        img = self.key_transform_map[key](img)
                        imgs.append(img)

                    # Process obs
                    img = obs_dict[key]
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
                        img = img.reshape(
                            batch_size * horizon, *img.shape[2:]
                        )  # (B*L,C,H,W)
                    assert (
                        img.shape[1:] == self.key_shape_map[key]
                    ), f"{img.shape[1:]} != {self.key_shape_map[key]}"  # (C,H,W)
                    img = self.key_transform_map[key](img)
                    imgs.append(img)

                # (N*B*L,C,H,W)
                imgs = torch.cat(imgs, dim=0)
                # (N*B*L,D)
                feature = self.key_model_map["rgb"](imgs)
                if "rgb" in self.feature_shape_map:
                    self.feature_shape_map["rgb"] = feature.shape[-1]
                # (N,B*L,D)
                if self.use_ep_first_obs and ep_first_obs_dict is not None:
                    feature = feature.reshape(
                        -1, batch_size * (horizon + 1), *feature.shape[1:]
                    )
                else:
                    feature = feature.reshape(
                        -1, batch_size * horizon, *feature.shape[1:]
                    )
                if horizon > 1:
                    if self.use_ep_first_obs and ep_first_obs_dict is not None:
                        # (N,B,L+1,D)
                        feature = feature.reshape(
                            -1, batch_size, horizon + 1, *feature.shape[2:]
                        )
                    else:
                        # (N,B,L,D)
                        feature = feature.reshape(
                            -1, batch_size, horizon, *feature.shape[2:]
                        )
                # (B,N,D) or (B,N,L,D)
                feature = torch.moveaxis(feature, 0, 1)
                if img_fea_only:
                    if horizon > 1:  # (B,L,N,D)
                        feature = torch.moveaxis(feature, 1, 2)
                        feature = feature.reshape(batch_size, horizon, -1)
                    else:
                        feature = feature.reshape(batch_size, -1)
                    return feature
                # (B,N*D) or (B,N*L*D)
                feature = feature.reshape(batch_size, -1)
                features.append(feature)
            else:
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
                    features.append(feature)

        # process pcd input
        if self.use_pcd_model and len(self.pcd_keys):
            pcd_obs_dict = {k: v for k, v in obs_dict.items() if k in self.pcd_keys}
            for key in pcd_obs_dict:
                if batch_size is None:
                    batch_size = pcd_obs_dict[key].shape[0]
                pcd_obs_dict[key] = pcd_obs_dict[key].reshape(
                    batch_size * horizon, *pcd_obs_dict[key].shape[2:]
                )
            feature = self.key_model_map["pcd"](pcd_obs_dict)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            # Process ep_first_obs
            if self.use_ep_first_obs and ep_first_obs_dict is not None:
                data = ep_first_obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                assert (
                    data.shape[2:] == self.key_shape_map[key]
                ), f"{data.shape}, {self.key_shape_map[key]}"  # bs, horizon
                data = data.reshape(batch_size, -1)
                features.append(data)
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert (
                data.shape[2:] == self.key_shape_map[key]
            ), f"{data.shape}, {self.key_shape_map[key]}"  # bs, horizon
            data = data.reshape(batch_size, -1)
            features.append(data)

        # concatenate all features
        result = torch.cat(
            features, dim=-1
        )  # B, feature_size (feature_size = resenet_fea+low_dim)

        if self.output_mlp is not None:
            result = self.output_mlp(result)
        elif self.output_vae is not None:
            mu, log_sigma = self.output_vae[0](result), self.output_vae[1](result)
            sigma = torch.exp(log_sigma)
            return torch.distributions.Normal(loc=mu, scale=sigma)

        return result  # B, feature_size

    @torch.no_grad()
    def output_shape(self, img_fea_only=False):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        horizon = self.n_obs_steps
        example_ep_first_obs_dict = None
        for key, attr in obs_shape_meta.items():
            shape = self.key_shape_map[key]
            this_obs = torch.zeros(
                (batch_size, horizon) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        if self.use_ep_first_obs:
            example_ep_first_obs_dict = dict()
            for key, attr in obs_shape_meta.items():
                shape = self.key_shape_map[key]
                this_obs = torch.zeros(
                    (batch_size,) + shape, dtype=self.dtype, device=self.device
                )
                example_ep_first_obs_dict[key] = this_obs

        example_output = self.forward(
            example_obs_dict, example_ep_first_obs_dict, img_fea_only=img_fea_only
        )
        output_shape = list(example_output.shape[1:])
        if len(output_shape) == 1:
            output_shape = output_shape[0]
        return output_shape
