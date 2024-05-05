import os

import torch
import torch.nn as nn

from .config.object_config import get_config as get_object_config
from .config.target_config import get_config as get_target_config
from .equinet import SE3ManiNet, SE3SegNet
from .utils import modified_gram_schmidt


class SE3PosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(__file__)
        self.target_cfg = get_target_config()
        self.object_cfg = get_object_config()

        self.policy_seg_target, self.policy_mani_target = self.build_se3_networks(
            base_dir, self.target_cfg
        )
        self.policy_seg_object, self.policy_mani_object = self.build_se3_networks(
            base_dir, self.object_cfg
        )

    def build_se3_networks(self, base_dir, cfg):
        cfg_seg, cfg_mani = cfg.seg, cfg.mani

        policy_seg = SE3SegNet(
            voxel_size=cfg_seg.voxel_size, radius_threshold=cfg_seg.radius_threshold
        ).float()
        policy_seg.load_state_dict(
            torch.load(os.path.join(base_dir, cfg_seg.resume_from))
        )
        policy_seg.eval()

        policy_mani = SE3ManiNet(
            voxel_size=cfg_mani.voxel_size,
            radius_threshold=cfg_mani.radius_threshold,
            feature_point_radius=cfg_mani.feature_point_radius,
        ).float()
        policy_mani.load_state_dict(
            torch.load(os.path.join(base_dir, cfg_mani.resume_from))
        )
        policy_mani.eval()

        return policy_seg, policy_mani

    def forward(self, xyz: torch.Tensor, rgb: torch.Tensor):
        ref_point = self.policy_seg_target({"xyz": xyz, "rgb": rgb})
        pred_target_pos, pred_target_direction = self.policy_mani_target(
            {"xyz": xyz, "rgb": rgb},
            reference_point=ref_point,
            distance_threshold=self.target_cfg.mani.distance_threshold,
        )
        pred_target_direction = modified_gram_schmidt(
            pred_target_direction.reshape(-1, 3).T, to_cuda=True
        )

        ref_point = self.policy_seg_object({"xyz": xyz, "rgb": rgb})
        pred_object_pos, pred_object_direction = self.policy_mani_object(
            {"xyz": xyz, "rgb": rgb},
            reference_point=ref_point,
            distance_threshold=self.object_cfg.mani.distance_threshold,
        )
        pred_object_direction = modified_gram_schmidt(
            pred_object_direction.reshape(-1, 3).T, to_cuda=True
        )
        return (
            pred_target_pos,
            pred_target_direction,
            pred_object_pos,
            pred_object_direction,
        )
