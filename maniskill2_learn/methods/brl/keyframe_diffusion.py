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
from maniskill2_learn.utils.torch import get_mean_lr, get_cuda_info, build_optimizer
from maniskill2_learn.utils.diffusion.helpers import Losses, apply_conditioning, cosine_beta_schedule, extract
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.diffusion.progress import Progress, Silent
from maniskill2_learn.utils.diffusion.mask_generator import LowdimMaskGenerator
from maniskill2_learn.utils.diffusion.normalizer import LinearNormalizer
from . import DiffAgent
from .keyframe_gpt import KeyframeGPTWithHist

from ..builder import BRL


@BRL.register_module()
class KeyDiffAgent(DiffAgent):
    def __init__(
        self,
        actor_cfg,
        keyframe_model_cfg,
        visual_nn_cfg,
        diff_nn_cfg,
        optim_cfg,
        env_params,
        action_seq_len,
        eval_action_len=1,
        pcd_cfg=None,
        lr_scheduler_cfg=None,
        batch_size=128,
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
        train_keyframe_model=True,
        train_diff_model=True,
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
        
        self.keyframe_model = KeyframeGPTWithHist(keyframe_model_cfg, keyframe_model_cfg.state_dim, keyframe_model_cfg.action_dim)

        self.train_keyframe_model = train_keyframe_model
        self.train_diff_model = train_diff_model

        if self.train_keyframe_model:
            self.keyframe_optim = self.keyframe_model.configure_adamw_optimizers()
                
        if not train_diff_model:
            self.actor_optim = None

        self.max_horizon = self.action_seq_len

    def keyframe_loss(self, states, timesteps, actions, keyframes, keytime_differences, keyframe_masks):
        keytime_differences /= self.max_horizon

        gt = torch.cat([keyframes, keytime_differences.unsqueeze(-1)], dim=-1) # (B, max_key_frame_len, act_dim+1)
        pred_keyframe, info = self.keyframe_model(states, timesteps, actions) # (B, future_seq_len, act_dim+1)
        loss = ((pred_keyframe[:,:keyframes.shape[1]] - gt) ** 2)
        masked_loss = loss*keyframe_masks.unsqueeze(-1)

        return masked_loss.mean(), info

    def diff_loss(self, x, masks, cond_mask, local_cond=None, global_cond=None, returns=None):
        # x is the action, with shape (bs, horizon, act_dim)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(x, t, cond_mask, local_cond, global_cond, returns)
        diffuse_loss = (diffuse_loss * masks.unsqueeze(-1)).mean()
        return diffuse_loss, info       

    def forward(self, observation, returns_rate=0.9, mode="eval", *args, **kwargs):

        # if mode=="eval": # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()
        
        observation = to_torch(observation, device=self.device, dtype=torch.float32)
        
        action_history = observation["actions"]
        action_history = self.normalizer.normalize(action_history)
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")
        
        self.set_mode(mode=mode)

        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask

        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask = self.mask_generator((bs, self.horizon, self.action_dim), self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
            else:
                raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")

        if act_mask.shape[0] < bs:
            act_mask = act_mask.repeat(max(bs//act_mask.shape[0]+1, 2), 1, 1)
        if act_mask.shape[0] != bs:
            act_mask = act_mask[:action_history.shape[0]] # obs mask is int
        
        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:,obs_mask,...]
        
        obs_fea = self.obs_encoder(observation) # No need to mask out since the history is set as the desired length
        
        if self.action_seq_len-hist_len:
            supp = torch.zeros(
                bs, self.action_seq_len-hist_len, self.action_dim, 
                dtype=action_history.dtype,
                device=self.device,
            )
            action_history = torch.concat([action_history, supp], dim=1)

        pred_action_seq = self.conditional_sample(cond_data=action_history, cond_mask=act_mask, global_cond=obs_fea, *args, **kwargs)
        pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
        pred_action = pred_action_seq

        if mode=="eval":
            pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len):,:]
            # Only used for ms-skill challenge online evaluation
            # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),:]
            # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            #     for i in range(self.eval_action_len-1):
            #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,:])
        
        return pred_action
    
    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            data = memory.get_all("actions")
            self.normalizer.fit(data, last_n_dims=1, mode='limits')
            self.init_normalizer = True

        batch_size = self.batch_size
        sampled_batch = memory.sample(batch_size)
        sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True) # ["obs","actions"]
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        if self.actor_optim is not None:
            self.actor_optim.zero_grad()
        if self.keyframe_optim is not None:
            self.keyframe_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)], 
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1), 
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        loss = 0.
        ret_dict = {}
        if self.train_diff_model:
            # generate impainting mask
            traj_data = sampled_batch["actions"]
            act_mask, obs_mask = None, None
            if self.fix_obs_steps:
                act_mask, obs_mask = self.act_mask, self.obs_mask
            if act_mask is None or obs_mask is None:
                if self.obs_as_global_cond:
                    act_mask, obs_mask = self.mask_generator(traj_data.shape, self.device)
                    self.act_mask, self.obs_mask = act_mask, obs_mask
                else:
                    raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")
            
            masked_obs = sampled_batch['obs']
            for key in masked_obs:
                masked_obs[key] = masked_obs[key][:,obs_mask,...]

            obs_fea = self.obs_encoder(masked_obs)
            traj_data = self.normalizer.normalize(traj_data)

            diff_loss, info = self.diff_loss(x=traj_data, masks=sampled_batch["is_valid"], cond_mask=act_mask, global_cond=obs_fea) # TODO: local_cond, returns
            ret_dict.update(info)
            loss += diff_loss
        if self.train_keyframe_model:
            keyframes = sampled_batch["keyframes"]
            keytime_differences = sampled_batch["keytime_differences"]
            keyframe_masks = sampled_batch["keyframe_masks"]

            timesteps = sampled_batch["timesteps"]
            observations = sampled_batch["obs"]
            states = observations["state"]
            actions = sampled_batch["actions"]
            keyframe_loss, info = self.keyframe_loss(states, timesteps, actions, keyframes, keytime_differences, keyframe_masks)
            ret_dict.update(info)
            loss += keyframe_loss
        
        loss.backward()
        if self.actor_optim is not None:
            self.actor_optim.step()
        if self.keyframe_optim is not None:
            self.keyframe_optim.step()

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
    
    def compute_test_loss(self, memory):
        logger = get_logger()
        logger.info(f"Begin to compute test loss with batch size {self.batch_size}!")
        ret_dict = {}
        num_samples = 0

        from maniskill2_learn.utils.meta import TqdmToLogger
        from tqdm import tqdm

        tqdm_obj = tqdm(total=memory.data_size, file=TqdmToLogger(), mininterval=20)

        batch_size = self.batch_size
        for sampled_batch in memory.mini_batch_sampler(self.batch_size, drop_last=False):
            sampled_batch = sampled_batch.to_torch(device="cuda", dtype="float32", non_blocking=True) # ["obs","actions"]

            is_valid = sampled_batch["is_valid"].squeeze(-1)
            loss, print_dict = self.compute_regression_loss(*sampled_batch)
            for key in print_dict:
                ret_dict[key] = ret_dict.get(key, 0) + print_dict[key] * len(sampled_batch)
            num_samples += len(sampled_batch)
            tqdm_obj.update(len(sampled_batch))

        logger.info(f"We compute the test loss over {num_samples} samples!")

        print_dict = {}
        print_dict["memory"] = get_total_memory("G", False)
        print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))
        print_info = dict_to_str(print_dict)
        logger.info(print_info)

        for key in ret_dict:
            ret_dict[key] /= num_samples

        return ret_dict