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

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z


@BRL.register_module()
class LatentDiffAgent(DiffAgent):
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

        self.max_horizon = self.action_seq_len - self.n_obs_steps # range [0, self.action_seq_len - self.n_obs_steps-1]

    def keyframe_loss(self, states, timesteps, actions, keyframe_states, keyframe_actions, keytime_differences, keyframe_masks):
        keytime_differences /= self.max_horizon

        gt_actions = torch.cat([keyframe_actions, keytime_differences.unsqueeze(-1)], dim=-1) # (B, max_key_frame_len, act_dim+1)
        gt_states = keyframe_states # (B, max_key_frame_len, state_dim)
        pred_keyframe_states, pred_keyframe_actions, info = self.keyframe_model(states, timesteps, actions) # (B, future_seq_len, act_dim+1)
        act_loss = ((pred_keyframe_actions[:,:keyframe_actions.shape[1]] - gt_actions) ** 2).sum(-1)
        state_loss = ((pred_keyframe_states[:,:keyframe_states.shape[1]] - gt_states) ** 2).sum(-1)

        masked_act_loss = act_loss*keyframe_masks
        masked_state_loss = state_loss*keyframe_masks

        masked_loss = masked_act_loss.sum(-1).mean() + masked_state_loss.sum(-1).mean()

        info.update(dict(keyframe_loss=masked_loss.item()))

        return masked_loss, info

    def diff_loss(self, x, masks, cond_mask, local_cond=None, global_cond=None, returns=None):
        # x is the action, with shape (bs, horizon, act_dim)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(x, t, cond_mask, local_cond, global_cond, returns)
        # diffuse_loss = (diffuse_loss * masks.unsqueeze(-1)).mean()
        diffuse_loss = diffuse_loss.mean()
        return diffuse_loss, info       
    
    def forward(self, observation, returns_rate=0.9, mode="eval", *args, **kwargs):

        # if mode=="eval": # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()
        
        observation = to_torch(observation, device=self.device, dtype=torch.float32)
        
        states = observation["state"]
        timesteps = observation["timesteps"]
        action_history = observation["actions"]
        action_history = self.normalizer.normalize(action_history)
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")
        
        self.set_mode(mode=mode)

        pred_keyframe_states, pred_keyframe_actions, info = self.keyframe_model(states, timesteps, action_history)
        pred_keyframe = pred_keyframe_actions[:, 0] # take the first key frame for diffusion
        pred_keyframe, pred_keytime_differences = pred_keyframe[:,:-1], pred_keyframe[:,-1] # split keyframe and predicted timestep
        pred_keytime_differences = pred_keytime_differences.cpu().numpy()
        # pred_keytime_differences = np.around(self.max_horizon * pred_keytime_differences, decimals=0)
        pred_keytime_differences = np.ceil(self.max_horizon * pred_keytime_differences)
        pred_keytime_differences = np.clip(pred_keytime_differences.astype(int), a_min=0, a_max=None)

        # Method 2: If bigger than max_horizon, then return keyframe util the keyframe is inside the prediction
        # if pred_keytime_differences[0] > self.max_horizon and mode == "eval": # Only support batch size = 1
        #     return self.normalizer.unnormalize(pred_keyframe.unsqueeze(1)) # [B, len, action_dim]
        
        # Method 1: Clip the difference to be in the range of max_horizon
        # pred_keytime_differences = np.clip(pred_keytime_differences, a_min=0, a_max=self.max_horizon) # [B,]
        pred_keytime_differences = np.clip(pred_keytime_differences, a_min=0, a_max=None) # [B,]
        
        pred_keytime = pred_keytime_differences + self.n_obs_steps - 1

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
            act_mask = act_mask[:action_history.shape[0]]
        
        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:,obs_mask,...] # obs mask is for one dimension
        
        obs_fea = self.obs_encoder(observation) # No need to mask out since the history is set as the desired length
        
        if self.action_seq_len-hist_len:
            supp = torch.zeros(
                bs, self.action_seq_len-hist_len, self.action_dim, 
                dtype=action_history.dtype,
                device=self.device,
            )
            action_history = torch.concat([action_history, supp], dim=1)
        if self.n_obs_steps < pred_keytime_differences[0] <= self.max_horizon and pred_keytime_differences[0] > 0: # Method3: only set key frame when less than horizon
        # if 0 < pred_keytime_differences[0] <= self.max_horizon: # Method3: only set key frame when less than horizon
            action_history[range(bs),pred_keytime] = pred_keyframe 
            act_mask = act_mask.clone()
            act_mask[range(bs),pred_keytime] = True

        # Predict action seq based on key frames
        pred_action_seq = self.conditional_sample(cond_data=action_history, cond_mask=act_mask, global_cond=obs_fea, *args, **kwargs)
        pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
        pred_action = pred_action_seq

        if mode=="eval":
            pred_action = pred_action_seq[:,hist_len:,:]
            # # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len):,:]
            # if self.n_obs_steps//2 < pred_keytime_differences[0] <= self.max_horizon: # Method3: only set key frame when less than horizon
            # # if 0 < pred_keytime_differences[0] <= self.max_horizon: # Method3: only set key frame when less than horizon
            #     # print("keyframe", timesteps[0,-1,0], pred_keytime_differences, self.normalizer.unnormalize(pred_keyframe), pred_action_seq[:,pred_keytime[0],:])
            #     # pred_action = pred_action_seq[:,hist_len:pred_keytime[0]+1,:] # do not support batch evaluation
            #     pred_action = pred_action_seq[:,hist_len:,:] # do not support batch evaluation
            # else:
            #     # print("no keyframe", timesteps[0,-1,0], pred_keytime_differences, self.normalizer.unnormalize(pred_keyframe))
            #     pred_action = pred_action_seq[:,hist_len:hist_len+4,:] # do not support batch evaluation
            #     # pred_action = pred_action_seq[:,hist_len:,:] # do not support batch evaluation
            # # Only used for ms-skill challenge online evaluation
            # # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),:]
            # # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            # #     for i in range(self.eval_action_len-1):
            # #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,:])
        
        return pred_action
    
    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            data = memory.get_all("actions")
            self.normalizer.fit(data, last_n_dims=1, mode='limits', range_eps=1e-7)
            self.init_normalizer = True

        batch_size = self.batch_size
        sampled_batch = memory.sample(batch_size, device=self.device, obs_mask=self.obs_mask, require_mask=True, action_normalizer=self.normalizer)
        # sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True) # ["obs","actions"] # Did in replay buffer
        
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
        
        # generate impainting mask
        traj_data = sampled_batch["actions"] # Need Normalize! (Already did in replay buffer)
        # traj_data = self.normalizer.normalize(traj_data)
        masked_obs = sampled_batch['obs']
        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask = self.mask_generator(traj_data.shape, self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
                for key in masked_obs:
                    masked_obs[key] = masked_obs[key][:,obs_mask,...]
            else:
                raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")

        if self.train_diff_model:
            obs_fea = self.obs_encoder(masked_obs)

            diff_loss, info = self.diff_loss(x=traj_data, masks=sampled_batch["is_valid"], cond_mask=act_mask, global_cond=obs_fea) # TODO: local_cond, returns
            ret_dict.update(info)
            loss += diff_loss

        if self.train_keyframe_model:
            keyframe_actions = sampled_batch["keyframe_actions"] # Need Normalize! (Already did in replay buffer)
            keyframe_states = sampled_batch["keyframe_states"]
            # keyframes = self.normalizer.normalize(keyframes)
            keytime_differences = sampled_batch["keytime_differences"]
            keyframe_masks = sampled_batch["keyframe_masks"]

            timesteps = sampled_batch["timesteps"]
            states = masked_obs["state"]
            actions = traj_data[:,obs_mask,...]
            keyframe_states = keyframe_states[:,obs_mask,...][:,-1] # We only take the last step of the horizon since we want to train the key frame model
            keyframe_actions = keyframe_actions[:,obs_mask,...][:,-1]
            keytime_differences = keytime_differences[:,obs_mask,...][:,-1]
            keyframe_masks = keyframe_masks[:,obs_mask,...][:,-1]

            keyframe_loss, info = self.keyframe_loss(states, timesteps, actions, keyframe_states, keyframe_actions, keytime_differences, keyframe_masks)
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