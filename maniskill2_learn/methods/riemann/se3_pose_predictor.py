import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .equinet import SE3SegNet, SE3ManiNet
from .utils import modified_gram_schmidt


class SE3PosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(__file__)
        all_cfg = OmegaConf.load(os.path.join(base_dir, "config/config.json"))
        self.cfg_seg = all_cfg.seg
        self.cfg_mani = all_cfg.mani

        self.policy_seg = SE3SegNet(
            voxel_size=self.cfg_seg.voxel_size, radius_threshold=self.cfg_seg.radius_threshold
        ).float()
        self.policy_seg.load_state_dict(torch.load(os.path.join(base_dir, self.cfg_seg.params_path)))
        self.policy_seg.eval()

        self.policy_mani = SE3ManiNet(
            voxel_size=self.cfg_mani.voxel_size, radius_threshold=self.cfg_mani.radius_threshold
        ).float()
        self.policy_mani.load_state_dict(torch.load(os.path.join(base_dir, self.cfg_mani.params_path)))
        self.policy_mani.eval()

    def forward(self, xyz: torch.Tensor, rgb: torch.Tensor):
        ref_point = self.policy_seg({"xyz": xyz, "rgb": rgb})
        pred_pos, pred_direction = self.policy_mani(
            {"xyz": xyz, "rgb": rgb},
            reference_point=ref_point,
            distance_threshold=self.cfg_mani.distance_threshold,
        )
        pred_direction = modified_gram_schmidt(
            pred_direction.reshape(-1, 3).T, to_cuda=True
        )
        return pred_pos, pred_direction
