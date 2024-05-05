import copy
from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch import nn

from maniskill2_learn.networks.builder import MODELNETWORKS
from maniskill2_learn.utils.diffusion.torch import replace_submodules
from maniskill2_learn.utils.torch import ExtendedModule


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Decoder(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x


@MODELNETWORKS.register_module()
class ImageObsDecoder(ExtendedModule):
    def __init__(
        self,
        shape_meta: dict,
        rgb_decoder_model: Union[nn.Module, Dict[str, nn.Module]] = ResNet18Decoder(),
        share_rgb_model: bool = True,
        use_group_norm: bool = False,
        n_obs_steps: int = 1,
        input_hidden_dim: int = 128,
    ):
        super().__int__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_shape_map = dict()

        obs_shape_meta = shape_meta["obs"]
        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_decoder_model, nn.Module)
            key_model_map["rgb"] = rgb_decoder_model

        # init key_model_map & key_transform_map
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            if isinstance(shape, list):
                shape = tuple(shape)
            if isinstance(shape, int):
                if "state" in key:
                    shape -= 9  # NOTE: We remove the dimension of velocity
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
                    if isinstance(rgb_decoder_model, dict):
                        # have provided model for each key
                        this_model = rgb_decoder_model[key]
                    else:
                        assert isinstance(rgb_decoder_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_decoder_model)

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
