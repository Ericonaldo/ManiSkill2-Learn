"""
Diffusion Policy
"""
from random import sample
from itertools import chain
from tqdm import tqdm
import numpy as np
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from maniskill2_learn.networks import build_model, build_reg_head
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.data import DictArray, GDict, dict_to_str
from maniskill2_learn.utils.meta import get_total_memory, get_logger
from maniskill2_learn.utils.torch import BaseAgent, get_mean_lr, get_cuda_info, build_optimizer
from maniskill2_learn.utils.diffusion.helpers import Losses, apply_conditioning, cosine_beta_schedule, extract
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.diffusion.progress import Progress, Silent
from maniskill2_learn.utils.diffusion.mask_generator import LowdimMaskGenerator
from maniskill2_learn.utils.diffusion.normalizer import LinearNormalizer
# from maniskill2_learn.networks.modules.multi_image_obs_encoder import MultiImageObsEncoder

from ..builder import BRL


@BRL.register_module()
class PromptDiffAgent(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        visual_nn_cfg,
        nn_cfg,
        optim_cfg,
        env_params,
        action_seq_len,
        eval_action_len=1,
        pcd_cfg=None,
        lr_scheduler_cfg=None,
        batch_size=256,
        n_timesteps=150,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        returns_condition=False,
        condition_guidance_w=0.1,
        agent_share_noise=False,
        obs_as_global_cond=True, # diffuse action or take obs as condition inputs
        action_visible=True, # If we cond on some hist actions
        fix_obs_steps=True, # Randomly cond on certain obs steps or deterministicly
        n_obs_steps=3,
        normalizer=LinearNormalizer(),
        **kwargs,
    ):
        super().__init__(
            actor_cfg=actor_cfg,
            visual_nn_cfg=visual_nn_cfg,
            nn_cfg=diff_nn_cfg,
            optim_cfg=optim_cfg,
            env_params=env_params,
            action_seq_len=action_seq_len,
            eval_action_len=eval_action_len,
            pcd_cfg=pcd_cfg,
            lr_scheduler_cfg=lr_scheduler_cfg,
            batch_size=batch_size,
            n_timesteps=n_timesteps,
            loss_type=loss_type,
            clip_denoised=clip_denoised,
            predict_epsilon=predict_epsilon,
            action_weight=action_weight,
            loss_discount=loss_discount,
            loss_weights=loss_weights,
            returns_condition=returns_condition,
            condition_guidance_w=condition_guidance_w,
            agent_share_noise=agent_share_noise,
            obs_as_global_cond=obs_as_global_cond,
            action_visible=action_visible,
            fix_obs_steps=fix_obs_steps,
            n_obs_steps=n_obs_steps,
            normalizer=normalizer,
        )
    
    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            data = memory.get_all("actions")
            self.normalizer.fit(data, last_n_dims=1, mode='limits')
            self.init_normalizer = True

        batch_size = self.batch_size
        sampled_batch = memory.sample(batch_size, device=self.device, obs_mask=self.obs_mask, require_mask=True, action_normalizer=self.normalizer)
        # sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True) # ["obs","actions"] # Did in replay buffer
        sampled_demo =  memory.sample(1, , device=self.device, obs_mask=self.obs_mask, require_mask=True, action_normalizer=self.normalizer, whole_traj=True)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        self.actor_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)], 
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1), 
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        # generate impainting mask
        traj_data = sampled_batch["actions"] # Need Normalize! (Already did in replay buffer)
        # traj_data = self.normalizer.normalize(traj_data)
        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask = self.mask_generator(traj_data.shape, self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
            else:
                raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")
        
        masked_obs_dict = sampled_batch['obs']
        for key in masked_obs_dict:
            masked_obs_dict[key] = masked_obs_dict[key][:,obs_mask,...]
        masked_obs_dict["demo"] = sampled_demo["obs"]

        act_dict = dict(
            "actions": sampled_batch['actions'],
            "demo": sampled_demo["actions"]
        )

        obs_fea = self.obs_encoder(masked_obs_dict, act_dict)

        loss, ret_dict = self.loss(x=traj_data, masks=sampled_batch["is_valid"], cond_mask=act_mask, global_cond=obs_fea) # TODO: local_cond, returns
        loss.backward()
        self.actor_optim.step()

        ## Not implement yet
        # if self.step % self.update_ema_every == 0:
        #     self.step_ema()
        ret_dict["grad_norm_diff_model"] = np.mean([torch.linalg.norm(parameter.grad.data).item() for parameter in self.model.parameters() if parameter.grad is not None])
        ret_dict["grad_norm_diff_obs_encoder"] = np.mean([torch.linalg.norm(parameter.grad.data).item() for parameter in self.obs_encoder.parameters() if parameter.grad is not None])

        if self.lr_scheduler is not None:
            ret_dict["lr"] = get_mean_lr(self.actor_optim)
        ret_dict = dict(ret_dict)
        ret_dict = {'diffusion/' + key: val for key, val in ret_dict.items()}
        
        self.step += 1

        return ret_dict