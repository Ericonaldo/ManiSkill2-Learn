from typing import Optional, Sequence

import torch
from torch import nn

from maniskill2_learn.utils.torch import ExtendedModule


def get_intersection_slice_mask(
    shape: tuple, dim_slices: Sequence[slice], device: Optional[torch.device] = None
):
    assert len(shape) == len(dim_slices)
    mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
    mask[dim_slices] = True
    return mask


def get_union_slice_mask(
    shape: tuple, dim_slices: Sequence[slice], device: Optional[torch.device] = None
):
    assert len(shape) == len(dim_slices)
    mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
    for i in range(len(dim_slices)):
        this_slices = [slice(None)] * len(shape)
        this_slices[i] = dim_slices[i]
        mask[this_slices] = True
    return mask


class DummyMaskGenerator(ExtendedModule):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, shape):
        device = self.device
        mask = torch.ones(size=shape, dtype=torch.bool, device=device)
        return mask


class LowdimMaskGenerator(ExtendedModule):
    def __init__(
        self,
        action_dim,
        obs_dim,
        # obs mask setup
        max_n_obs_steps=3,
        # action mask
        action_visible=True,
        return_one_mask=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.action_visible = action_visible
        self.return_one_mask = return_one_mask

    @torch.no_grad()
    def forward(self, shape, device, seed=None):
        # device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., -self.action_dim :] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim

        mask = None
        if self.return_one_mask:
            mask = obs_mask & is_obs_dim
            if self.action_visible:
                mask = mask | action_mask

            # return mask
        if self.obs_dim <= 0:
            obs_mask = obs_mask[0, :, 0]
        return action_mask, obs_mask, mask


class KeyframeMaskGenerator(ExtendedModule):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        # obs mask setup
        max_n_obs_steps: int = 3,
        # action mask
        action_visible: bool = True,
        return_one_mask: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.action_visible = action_visible
        self.return_one_mask = return_one_mask

    @torch.no_grad()
    def forward(self, shape, device, seed=None):
        # device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., -self.action_dim :] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim
            # NOTE: the only difference
            action_mask[:, -1] = True

        mask = None
        if self.return_one_mask:
            mask = obs_mask & is_obs_dim
            if self.action_visible:
                mask = mask | action_mask

            # return mask
        if self.obs_dim <= 0:
            obs_mask = obs_mask[0, :, 0]
        return action_mask, obs_mask, mask
