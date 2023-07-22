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
from torch.nn.parallel import DistributedDataParallel as DDP

import sapien.core as sapien

from maniskill2_learn.networks import build_model, build_reg_head
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.data import DictArray, GDict, dict_to_str
from maniskill2_learn.utils.torch import load_state_dict
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

from transforms3d.quaternions import quat2axangle

def inv_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action` without clipping."""
    low, high = np.asarray(low), np.asarray(high)
    return (action - 0.5 * (high + low)) / (0.5 * (high - low))

def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta

def delta_pose_to_pd_ee_delta(
    delta_pose: sapien.Pose,
):
    delta_pose = np.r_[
        delta_pose.p,
        compact_axis_angle_from_quaternion(delta_pose.q),
    ]
    return inv_scale_action(delta_pose, -1, 1)

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
        diffuse_state=False,
        train_keyframe_model=True,
        train_diff_model=True,
        diffusion_updates=None,
        keyframe_model_updates=None,
        keyframe_model_path=None,
        use_keyframe=True,
        use_ep_first_obs=False,
        pred_keyframe_num=1,
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
            diffuse_state=diffuse_state,
        )
        
        self.keyframe_model = KeyframeGPTWithHist(keyframe_model_cfg, keyframe_model_cfg.state_dim, keyframe_model_cfg.action_dim)

        self.train_keyframe_model = train_keyframe_model
        self.train_diff_model = train_diff_model

        if self.train_keyframe_model:
            self.keyframe_optim = self.keyframe_model.configure_adamw_optimizers()
                
        if not train_diff_model:
            self.actor_optim = None

        self.max_horizon = self.action_seq_len - self.n_obs_steps # range [0, self.action_seq_len - self.n_obs_steps-1]

        self.diffusion_updates = diffusion_updates
        self.keyframe_model_updates = keyframe_model_updates
        self.use_keyframe = use_keyframe
        self.pred_keyframe_num = pred_keyframe_num
        self.use_ep_first_obs = use_ep_first_obs
        
        self.keyframe_model_path = keyframe_model_path
        if self.keyframe_model_path is not None:
            self.load_keyframe_model()

        self.last_state = None

    def load_keyframe_model(self):
        if self.keyframe_model_path is None:
            return
        print("loading keyframe model in {}".format(self.keyframe_model_path))
        loaded_dict = torch.load(self.keyframe_model_path, map_location=self.device)
        if not isinstance(self, DDP):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(loaded_dict['state_dict'], prefix="module.")
            if 'optimizer' in loaded_dict:
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(loaded_dict['optimizer'], prefix="module.")
        
        load_state_dict(self.keyframe_model, loaded_dict['state_dict'])
        if 'optimizer' in loaded_dict.keys():
            load_state_dict(self.keyframe_optim, loaded_dict['optimizer'])

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
        info.update({"action_diff_loss": diffuse_loss.item()})
        return diffuse_loss, info       
    
    def forward(self, observation, returns_rate=0.9, mode="eval", *args, **kwargs):

        # if mode=="eval": # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()
        
        observation = to_torch(observation, device=self.device, dtype=torch.float32)
        # if "state" in observation:
        #     observation["state"] = torch.cat([observation["state"][...,:9], observation["state"][...,18:]], axis=-1)
        
        states = observation["state"]
        timesteps = observation["timesteps"]
        action_history = observation["actions"]
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")
        
        self.set_mode(mode=mode)

        pred_keyframe_states, pred_keyframe_actions, info = self.keyframe_model(states, timesteps, action_history)
        pred_keyframe = pred_keyframe_actions[:, :self.pred_keyframe_num] # take the first key frame for diffusion
        pred_keyframe, pred_keytime_differences = pred_keyframe[:,:,:-1], pred_keyframe[:,:,-1] # split keyframe and predicted timestep
        if self.diffuse_state:
            pred_keyframe_states = pred_keyframe_states[:,:self.pred_keyframe_num]
            pred_keyframe = torch.cat([pred_keyframe_states, pred_keyframe], dim=-1)
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

        act_mask, obs_mask, data_mask = None, None, None
        if self.fix_obs_steps:
            act_mask, obs_mask, data_mask = self.act_mask, self.obs_mask, self.data_mask
        
        data_dim = self.action_dim+self.state_dim if self.diffuse_state else self.action_dim
                
        if data_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask, data_mask = self.mask_generator((bs, self.horizon, data_dim), self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
                if data_mask is None:
                    data_mask = act_mask
                self.data_mask = data_mask
                if self.diffuse_state:
                    self.obs_mask = obs_mask = obs_mask[0,:,0]
            else:
                raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")

        if data_mask.shape[0] < bs:
            data_mask = data_mask.repeat(max(bs//data_mask.shape[0]+1, 2), 1, 1)
        if data_mask.shape[0] != bs:
            data_mask = data_mask[:action_history.shape[0]]
        
        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:,obs_mask,...] # obs mask is for one dimension
        
        obs_fea = self.obs_encoder(observation) # No need to mask out since the history is set as the desired length
        
        data_history = action_history
        if self.action_seq_len-hist_len:
            if self.diffuse_state:
                action_to_state_horizon_supp = torch.zeros(
                    bs, 1, self.action_dim, 
                    dtype=action_history.dtype,
                    device=self.device,
                )
                action_history = torch.cat([action_history, action_to_state_horizon_supp], dim=1)
                data_history = torch.cat([states, action_history], dim=-1)
                supp = torch.zeros(
                    bs, self.action_seq_len-hist_len-1, data_dim, 
                    dtype=action_history.dtype,
                    device=self.device,
                )
            else:
                supp = torch.zeros(
                    bs, self.action_seq_len-hist_len, self.action_dim, 
                    dtype=action_history.dtype,
                    device=self.device,
                )
            data_history = torch.cat([data_history, supp], dim=1)
        
        data_history = self.normalizer.normalize(data_history)
        # print("before: ", data_mask[...,0])
        if self.use_keyframe:
            # pred_keyframe = self.normalizer.normalize(pred_keyframe)
            for i in range(len(pred_keytime_differences[0])):
                if self.n_obs_steps < pred_keytime_differences[0][i] <= self.max_horizon and pred_keytime_differences[0][i] > 0: # Method3: only set key frame when less than horizon
                # if 0 < pred_keytime_differences[0] <= self.max_horizon: # Method3: only set key frame when less than horizon
                    data_history[range(bs),pred_keytime[:,i:i+1]] = pred_keyframe[:,i:i+1]
                    data_mask = data_mask.clone()
                    data_mask[range(bs),pred_keytime[:,i:i+1],:-self.action_dim] = True
                    # data_mask[range(bs),pred_keytime[:,i],:] = True
                else:
                    break
        # print("after: ", data_mask[...,0], pred_keytime_differences)

        # Predict action seq based on key frames
        pred_action_seq = self.conditional_sample(cond_data=data_history, cond_mask=data_mask, global_cond=obs_fea, *args, **kwargs)
        pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
        pred_action = pred_action_seq

        if mode=="eval":
            pred_action = pred_action_seq[:,hist_len:,-self.action_dim:]

            # Check pose
            # if self.last_state is not None:
            #     last_pose = self.last_state[...,-1,-13:-6].cpu().numpy()
            #     cur_pose = observation['state'][...,-1,-13:-6].cpu().numpy()
            #     last_pose = sapien.Pose(p=last_pose[0][:3], q=last_pose[0][3:])
            #     cur_pose = sapien.Pose(p=cur_pose[0][:3], q=cur_pose[0][3:])
            #     delta_pose = last_pose.inv() * cur_pose
            #     delta_pose = delta_pose_to_pd_ee_delta(delta_pose)
            #     print(1, delta_pose)
            #     print(2, pred_action[0,hist_len-1:hist_len+3])
            #     print(3, pred_action_seq[0,hist_len-1:hist_len+3,-13:-6])
            # self.last_state = observation['state']

            # Compute pose
            # cur_tcq_pose_np = observation['state'][...,-1,-13:-6].cpu().numpy()
            # pred_next_tcq_pose_np = pred_action_seq[:,hist_len+4:,-self.action_dim-13:-self.action_dim-6].cpu().numpy()

            # pred_action = np.zeros((pred_action_seq.shape[1] - hist_len - 4, self.action_dim))
            # cur_tcq_pose = sapien.Pose(p=cur_tcq_pose_np[0][:3], q=cur_tcq_pose_np[0][3:])
            # for i in range(len(pred_action)):
            #     pred_next_tcq_pose = sapien.Pose(p=pred_next_tcq_pose_np[0][i][:3], q=pred_next_tcq_pose_np[0][i][3:])
            #     pred_delta_pose = cur_tcq_pose.inv() * pred_next_tcq_pose
            #     pred_delta_pose = delta_pose_to_pd_ee_delta(pred_delta_pose)
            #     pred_action[i, :-1] = pred_delta_pose
            # pred_action[:, -1] = pred_action_seq[0,hist_len+3:-1,-1].cpu().numpy()
            # pred_action = np.expand_dims(pred_action, axis=0)
            # print(1, pred_action)
            # print(2, pred_action_seq[:,hist_len:,-self.action_dim:])


            # # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len):,:]
            # if self.n_obs_steps < pred_keytime_differences[0][-1] <= self.max_horizon and pred_keytime_differences[0][-1] > 0: # Method3: only set key frame when less than horizon
            # # # if 0 < pred_keytime_differences[0] <= self.max_horizon: # Method3: only set key frame when less than horizon
            # #     # print("keyframe", timesteps[0,-1,0], pred_keytime_differences, self.normalizer.unnormalize(pred_keyframe), pred_action_seq[:,pred_keytime[0],:])
            # #     pred_action = pred_action_seq[:,hist_len:pred_keytime[0][-1]+1,-self.action_dim:] # do not support batch evaluation
            # # #     pred_action = pred_action_seq[:,hist_len:,:] # do not support batch evaluation
            # else:
            # #     # print("no keyframe", timesteps[0,-1,0], pred_keytime_differences, self.normalizer.unnormalize(pred_keyframe))
            #     pred_action = pred_action_seq[:,hist_len:hist_len+self.eval_action_len,-self.action_dim:] # do not support batch evaluation
            #     pred_action = pred_action_seq[:,hist_len:,-self.action_dim:] # do not support batch evaluation
            # # Only used for ms-skill challenge online evaluation
            # # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),:]
            # # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            # #     for i in range(self.eval_action_len-1):
            # #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,:])
        
        return pred_action
    
    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            action_data = memory.get_all("actions")
            # data = action_data
            obs_data = memory.get_all("obs", "state")
            if self.diffuse_state:
                data = np.concatenate([obs_data, action_data], axis=-1)
            else:
                data = action_data
            self.normalizer.fit(data, last_n_dims=1, mode='limits', range_eps=1e-7)
            self.init_normalizer = True

        batch_size = self.batch_size
        if self.diffuse_state:
            sampled_batch = memory.sample(batch_size, device=self.device, obs_mask=self.obs_mask, require_mask=True, obsact_normalizer=self.normalizer)
        else:
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

        ep_first_obs = None
        if self.use_ep_first_obs and 'ep_first_obs' in sampled_batch:
            ep_first_obs = sampled_batch['ep_first_obs']
        
        # generate impainting mask
        if self.diffuse_state:
            traj_data = torch.cat([sampled_batch["states"],sampled_batch["actions"]], dim=-1)
        else:
            traj_data = sampled_batch["actions"]
        # Need Normalize! (Already did in replay buffer)
        # traj_data = self.normalizer.normalize(traj_data)
        masked_obs = sampled_batch['obs']
        act_mask, obs_mask, data_mask = None, None, None
        if self.fix_obs_steps:
            act_mask, obs_mask, data_mask = self.act_mask, self.obs_mask, self.data_mask
        if data_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask, data_mask = self.mask_generator(traj_data.shape, self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
                if data_mask is None:
                    data_mask = act_mask
                self.data_mask = data_mask

                if self.diffuse_state:
                    self.obs_mask = obs_mask = obs_mask[0,:,0]

                for key in masked_obs:
                    masked_obs[key] = masked_obs[key][:,obs_mask,...]
                    
            else:
                raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")

        if self.train_diff_model:
            if (self.diffusion_updates is None) or ((self.diffusion_updates is not None) and updates > self.diffusion_updates):
                obs_fea = self.obs_encoder(masked_obs, ep_first_obs=ep_first_obs)

                diff_loss, info = self.diff_loss(x=traj_data, masks=sampled_batch["is_valid"], cond_mask=data_mask, global_cond=obs_fea) # TODO: local_cond, returns
                ret_dict.update(info)
                loss += diff_loss

        if self.train_keyframe_model:
            if (self.keyframe_model_updates is None) or ((self.keyframe_model_updates is not None) and updates > self.keyframe_model_updates):
                keyframe_actions = sampled_batch["keyframe_actions"] # Need Normalize! (Already did in replay buffer)
                keyframe_states = sampled_batch["keyframe_states"]
                # keyframes = self.normalizer.normalize(keyframes)
                keytime_differences = sampled_batch["keytime_differences"]
                keyframe_masks = sampled_batch["keyframe_masks"]

                timesteps = sampled_batch["timesteps"]
                states = masked_obs["state"]
                if ep_first_obs is not None: # Append ep first obs for predicting keyframes
                    states = torch.cat([ep_first_obs['state'].unsqueeze(1), states], dim=1)
                actions = sampled_batch["actions"][:,obs_mask,...]
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