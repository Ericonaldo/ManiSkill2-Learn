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
from maniskill2_learn.networks.modules.block_utils import SimpleMLP as MLP
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
        keyframe_state_only,
        keyframe_optim_cfg=None,
        eval_action_len=1,
        pcd_cfg=None,
        lr_scheduler_cfg=None,
        keyframe_lr_scheduler_cfg=None,
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
        use_keyframe=False,
        use_ep_first_obs=False,
        pred_keyframe_num=1,
        pose_only=False,
        keyframe_pose_only=False,
        keyframe_model_type="gpt",
        pose_dim=7,
        extra_dim=0,
        **kwargs,
    ):
        visual_nn_cfg.update(use_ep_first_obs = use_ep_first_obs)
        assert keyframe_model_type in ["gpt", "bc"]
        if keyframe_model_type == "bc":
            pose_only = True
        self.keyframe_model_type = keyframe_model_type
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
            pose_only=pose_only,
            pose_dim=pose_dim,
            extra_dim=extra_dim,
        )

        self.keyframe_pose_only = pose_only and keyframe_pose_only

        if keyframe_optim_cfg is None:
            keyframe_optim_cfg = optim_cfg
        self.keyframe_obs_encoder = build_model(visual_nn_cfg)
        
        if keyframe_model_type == "gpt":
            keyframe_state_dim = keyframe_model_cfg.state_dim
            if not keyframe_state_only:
                keyframe_state_dim += self.img_feature_dim
            self.keyframe_model = KeyframeGPTWithHist(keyframe_model_cfg, keyframe_state_dim, keyframe_model_cfg.action_dim, pred_state_dim=keyframe_model_cfg.state_dim, use_first_state=use_ep_first_obs, pose_only=keyframe_pose_only, pose_dim=pose_dim)
        elif keyframe_model_type == "bc":
            self.keyframe_model = MLP(input_dim=self.obs_feature_dim, output_dim=self.pose_dim+1, hidden_dims=[2048, 512, 128])
        else:
            raise NotImplementedError

        self.train_keyframe_model = train_keyframe_model
        self.train_diff_model = train_diff_model

        self.keyframe_optim = None
        if self.train_keyframe_model:
            if keyframe_model_type == "gpt":
                self.keyframe_optim = self.keyframe_model.configure_adamw_optimizers(extra_model=self.keyframe_obs_encoder)
            elif keyframe_model_type == "bc":
                self.keyframe_optim = build_optimizer([self.keyframe_obs_encoder,self.keyframe_model], keyframe_optim_cfg)
                self.actor_optim = build_optimizer([self.model,self.obs_encoder], optim_cfg) # Update using the same optimizer
        else:
            del self.keyframe_model
            del self.keyframe_obs_encoder
        
        if keyframe_lr_scheduler_cfg is None:
            self.keyframe_lr_scheduler = None
        else:
            if self.keyframe_optim is not None:
                keyframe_lr_scheduler_cfg["optimizer"] = self.keyframe_optim
            self.keyframe_lr_scheduler = build_lr_scheduler(keyframe_lr_scheduler_cfg)
                
        if not train_diff_model:
            self.actor_optim = None
            del self.model
            del self.obs_encoder

        if train_diff_model and train_keyframe_model and (not keyframe_state_only):
            assert id(self.obs_encoder.key_model_map["rgb"]) != id(self.keyframe_obs_encoder.key_model_map["rgb"]), "same encoder?????"

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
        self.keyframe_state_only = keyframe_state_only

    def load_keyframe_model(self):
        if self.keyframe_model_path is None:
            return
        print("loading keyframe model in {}".format(self.keyframe_model_path))
        loaded_dict = torch.load(self.keyframe_model_path, map_location=self.device)
        if not isinstance(self, DDP):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(loaded_dict['state_dict'], prefix="module.")
            if 'optimizer' in loaded_dict:
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(loaded_dict['optimizer'], prefix="module.")
        
        keyframe_state_dict = {_:loaded_dict['state_dict'][_] for _ in loaded_dict['state_dict'].keys() if "keyframe_model" in _}
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(keyframe_state_dict, prefix="keyframe_model.")
        load_state_dict(self.keyframe_model, keyframe_state_dict)

        keyframe_obs_encoder_state_dict = {_:loaded_dict['state_dict'][_] for _ in loaded_dict['state_dict'].keys() if "keyframe_obs_encoder" in _}
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(keyframe_obs_encoder_state_dict, prefix="keyframe_obs_encoder.")
        load_state_dict(self.keyframe_obs_encoder, keyframe_obs_encoder_state_dict)
        
        if 'optimizer' in loaded_dict.keys():
            load_state_dict(self.keyframe_optim, loaded_dict['optimizer'])

    def keyframe_bc_loss(self, states, keyframe_states, keytime_differences, keyframe_masks):
        keytime_differences /= self.max_horizon
        if self.extra_dim > 0:
            gt_states = torch.cat([keyframe_states[...,-self.pose_dim-self.extra_dim:-self.extra_dim], keytime_differences.unsqueeze(-1)], dim=-1) # (B, max_key_frame_len, self.pose_dim+1)
        else:
            gt_states = torch.cat([keyframe_states[...,-self.pose_dim:], keytime_differences.unsqueeze(-1)], dim=-1) # (B, max_key_frame_len, self.pose_dim+1)

        info={}
        pred_keyframe_states = self.keyframe_model(states) # We expect states are obs features, pred_keyframe_states [B, self.pose_dim]
        
        state_loss = ((pred_keyframe_states[:,:keyframe_states.shape[1]] - gt_states) ** 2).sum(-1)

        masked_state_loss = state_loss*keyframe_masks

        masked_loss = masked_state_loss.sum(-1).mean()

        info.update(dict(keyframe_loss=masked_loss.item()))

        return masked_loss, info

    def keyframe_gpt_loss(self, obs, timesteps, actions, keyframe_states, keyframe_actions, keytime_differences, keyframe_masks, ep_first_state=None):
        keytime_differences /= self.max_horizon

        if self.pose_only:
            if self.extra_dim > 0:
                gt_states = torch.cat([keyframe_states[...,-self.pose_dim-self.extra_dim:-self.extra_dim], keytime_differences.unsqueeze(-1)], dim=-1) # (B, max_key_frame_len, self.pose_dim+1)
            else:
                gt_states = keyframe_states[...,-self.pose_dim:] # (B, max_key_frame_len, self.pose_dim)
        else:
            gt_states = keyframe_states # (B, max_key_frame_len, state_dim)
        gt_actions = torch.cat([keyframe_actions, keytime_differences.unsqueeze(-1)], dim=-1) # (B, max_key_frame_len, act_dim+1)

        pred_keyframe_states, pred_keyframe_actions, info = self.keyframe_model(obs, timesteps, actions, first_state=ep_first_state) # (B, future_seq_len, act_dim+1)
        
        if False: # self.pose_only:
            state_loss = ((pred_keyframe_states[:,:gt_states.shape[1]] - gt_states) ** 2).sum(-1)
            masked_state_loss = state_loss*keyframe_masks

            masked_loss = masked_state_loss.sum(-1).mean()

        else:
            act_loss = ((pred_keyframe_actions[:,:keyframe_actions.shape[1]] - gt_actions) ** 2).sum(-1)
            state_loss = ((pred_keyframe_states[:,:gt_states.shape[1]] - gt_states) ** 2).sum(-1)

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
        

        # data = torch.cat([observation["state"], torch.cat([observation["actions"], observation["actions"][:,0:1,:]], dim=1)], dim=-1)
        # data = self.normalizer.normalize(data)
        # observation["state"] = data[..., :-self.action_dim]
        # observation["actions"] = data[..., :-1, -self.action_dim:]
        # states = data[..., :-self.action_dim]
        # action_history = data[..., :-1, -self.action_dim:]
        
        states = observation["state"].clone()
        action_history = observation["actions"]
        timesteps = observation["timesteps"]
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")
        observation.pop("timesteps")

        ep_first_obs = None
        ep_first_obs_state = None
        if self.use_ep_first_obs and ('ep_first_obs' in observation):
            ep_first_obs = observation['ep_first_obs']
            
            data = torch.cat([ep_first_obs['state'], torch.zeros((ep_first_obs['state'].shape[0], self.action_dim), dtype=states.dtype,device=self.device,)], dim=-1)
            data = self.normalizer.normalize(data)
            ep_first_obs['state'] = data[...,:-self.action_dim]
            ep_first_obs_state = ep_first_obs['state']
        observation.pop("ep_first_obs", None)
        
        self.set_mode(mode=mode)
        if self.use_keyframe:
            keyframe_inputs = states.clone()
            if not self.keyframe_state_only:
                tmp_observation = dict()
                for key in observation:
                    tmp_observation[key] = observation[key].clone()
                img_obs_fea = self.keyframe_obs_encoder(tmp_observation, ep_first_obs_dict=ep_first_obs, img_fea_only=True)
                keyframe_inputs = torch.cat([img_obs_fea, states.clone()], dim=-1)
            pred_keyframe_states, pred_keyframe_actions, info = self.keyframe_model(keyframe_inputs, timesteps, action_history.clone(), first_state=ep_first_obs_state)
            
            if pred_keyframe_actions is not None:
                pred_keyframe_actions, pred_keytime_differences = pred_keyframe_actions[:,:,:-1], pred_keyframe_actions[:,:,-1] # split keyframe and predicted timestep
                pred_keyframe = pred_keyframe_actions[:, :self.pred_keyframe_num] # take the first key frame for diffusion
                if self.diffuse_state:
                    if pred_keyframe_states.shape[-1] == self.pose_dim + 1: # Old problem, for compatibility
                        pred_keyframe_states, tmp_pred_keytime_differences = pred_keyframe_states[:,:,:-1], pred_keyframe_states[:,:,-1] # split keyframe and predicted timestep
                    pred_keyframe_states = pred_keyframe_states[:,:self.pred_keyframe_num]

                    if self.keyframe_pose_only and (pred_keyframe_states.shape[-1] < self.state_dim):
                        assert pred_keyframe_states.shape[-1] == self.pose_dim, "what are you predicting? pose should be {} but your prediction is {}".format(self.pose_dim, pred_keyframe_states.shape)
                        # pred_keyframe_states is only the pose
                        pred_keyframe_poses = pred_keyframe_states
                        extra_supp = torch.zeros(
                            pred_keyframe_poses.shape[0], pred_keyframe_poses.shape[1], self.extra_dim, 
                            dtype=action_history.dtype,
                            device=self.device,
                        )
                        before_pose_supp = torch.zeros(
                            pred_keyframe_poses.shape[0], pred_keyframe_poses.shape[1], self.state_dim-self.pose_dim-self.extra_dim, 
                            dtype=action_history.dtype,
                            device=self.device,
                        )
                        pred_keyframe_states = torch.cat([before_pose_supp, pred_keyframe_poses, extra_supp], dim=-1)
                    
                    if not self.keyframe_pose_only:
                        # Set the robot pose from the history
                        pred_keyframe_states[...,-self.extra_dim-self.pose_dim-self.action_dim-7:-self.extra_dim-self.pose_dim-self.action_dim] = states[:,-1:,-self.extra_dim-self.pose_dim-self.action_dim-7:-self.extra_dim-self.pose_dim-self.action_dim].clone()
                    # Compute the information of the extra dim
                    if self.extra_dim > 0:
                        assert self.extra_dim == 6, "extra dim should be 6!"
                        # states[...,-6:-3]-states[...,-13:-10] == states[...,-3:]
                        pred_keyframe_states[...,-6:-3] = states[0,0,-6:-3].clone() # Target pos
                        pred_keyframe_states[...,-3:] = states[0,0,-6:-3]-pred_keyframe_states[...,-13:-10] # Delta pos to predicted tcp
                        # print(pred_keyframe_states[...,:], "\n", states[...,:])
                        # print("==========")

                    pred_keyframe = torch.cat([pred_keyframe_states, pred_keyframe], dim=-1)
                        
                    # Norm the pred keyframe
                    pred_keyframe = self.normalizer.normalize(pred_keyframe)

                    # Only select the states as the keyframe
                    pred_keyframe_states, pred_keyframe_actions = pred_keyframe[..., :self.state_dim], pred_keyframe[..., -self.action_dim:]
                    
                    pred_keyframe = pred_keyframe_states

            else: # We have not checked this path
                raise NotImplementedError
                pred_keyframe = pred_keyframe_states[:, :self.pred_keyframe_num]
                pred_keyframe, pred_keytime_differences = pred_keyframe[:,:,:-1], pred_keyframe[:,:,-1] # split keyframe and predicted timestep
            
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
        
        if not self.diffuse_state:
            data_dim = self.action_dim
        else:
            data_dim = self.action_dim+self.pose_dim if self.pose_only else self.action_dim+self.state_dim
        # data_dim = self.action_dim+self.state_dim if self.diffuse_state else self.action_dim
                
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

        obs_fea = self.obs_encoder(observation, ep_first_obs_dict=ep_first_obs) # No need to mask out since the history is set as the desired length
        
        # data = torch.cat([observation["state"], torch.cat([observation["actions"], observation["actions"][:,0:1,:]], dim=1)], dim=-1)
        # data = torch.cat([states, torch.cat([action_history, action_history[:,0:1,:]], dim=1)], dim=-1)
        # data = self.normalizer.normalize(data)
        # states = data[..., :-self.action_dim]
        # action_history = data[..., :-1, -self.action_dim:]
        
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
                    bs, self.action_seq_len-hist_len-1, self.action_dim+self.state_dim, 
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

        if self.diffuse_state and self.pose_only:
            if self.extra_dim > 0:
                data_history = torch.cat([data_history[...,-self.action_dim-self.pose_dim-self.extra_dim:-self.action_dim-self.extra_dim], data_history[...,-self.action_dim:]], dim=-1)
            else:
                data_history = torch.cat([data_history[...,-self.action_dim-self.pose_dim:-self.action_dim], data_history[...,-self.action_dim:]], dim=-1)

        if self.use_keyframe:
            pred_keyframe = pred_keyframe[...,:self.state_dim]

            for i in range(self.pred_keyframe_num):
                if self.n_obs_steps < pred_keytime_differences[0][i] <= self.max_horizon: # Method3: only set key frame when less than horizon
                # if self.n_obs_steps < pred_keytime_differences[0]: 
                # if 0 < pred_keytime_differences[0,i] <= self.max_horizon: # Method3: only set key frame when less than horizon
                    # data_history[range(bs),pred_keytime[:,i:i+1]] = pred_keyframe[:,i:i+1]
                    # pred_keytime[:,i:i+1] = min(self.action_seq_len-1, pred_keytime[:,i:i+1])
                    # print(data_history[:,:6,:-self.action_dim], "\n 2:", pred_keyframe)

                    if self.keyframe_pose_only:
                        if self.extra_dim > 0:
                            data_history[range(bs),pred_keytime[:,i:i+1], -self.extra_dim-self.pose_dim-self.action_dim:-self.action_dim-self.extra_dim] = pred_keyframe[:,i:i+1][...,-self.extra_dim-self.pose_dim:-self.extra_dim]
                        else:
                            data_history[range(bs),pred_keytime[:,i:i+1], -self.pose_dim-self.action_dim:-self.action_dim] = pred_keyframe[:,i:i+1][...,-self.pose_dim:]
                        data_mask = data_mask.clone()
                        data_mask[range(bs),pred_keytime[:,i:i+1],-self.extra_dim-self.pose_dim-self.action_dim:-self.action_dim-self.extra_dim] = True
                    else:
                        if self.pose_only:
                            data_history[range(bs),pred_keytime[:,i:i+1], -self.pose_dim-self.action_dim-self.extra_dim:-self.action_dim-self.extra_dim] = pred_keyframe[:,i:i+1][...,-self.pose_dim:]
                        else:
                            data_history[range(bs),pred_keytime[:,i:i+1], :-self.action_dim] = pred_keyframe[:,i:i+1]
                        data_mask = data_mask.clone()
                        if self.diffuse_state:
                            data_mask[range(bs),pred_keytime[:,i],:-self.action_dim] = True
                        else:
                            data_mask[range(bs),pred_keytime[:,i],:] = True
                else:
                    break
        # print("after: ", data_mask[...,0], pred_keytime_differences)

        # Predict action seq based on key frames
        pred_action_seq = self.conditional_sample(cond_data=data_history, cond_mask=data_mask, global_cond=obs_fea, *args, **kwargs)
        data = pred_action_seq
        if (pred_action_seq.shape[2] != self.action_dim and pred_action_seq.shape[2]) != (self.action_dim+self.state_dim) :
            supp = torch.zeros(
                pred_action_seq.shape[0], pred_action_seq.shape[1], self.state_dim+self.action_dim-pred_action_seq.shape[2], 
                dtype=pred_action_seq.dtype,
                device=self.device,
            )
            data = torch.cat([supp, pred_action_seq], dim=-1)
        data = self.normalizer.unnormalize(data)
        pred_action_seq = data[...,-self.action_dim:]

        if mode=="eval":
            pred_action = pred_action_seq[:,hist_len:]

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
            # if self.use_keyframe:
            #     if self.n_obs_steps < pred_keytime_differences[0][-1] <= self.max_horizon and pred_keytime_differences[0][-1] > 0: # Method3: only set key frame when less than horizon
            #     # if 0 < pred_keytime_differences[0] <= self.max_horizon: # Method3: only set key frame when less than horizon
            #         # print("keyframe", timesteps[0,-1,0], pred_keytime_differences, self.normalizer.unnormalize(pred_keyframe), pred_action_seq[:,pred_keytime[0],:])
            #         # pred_action = pred_action_seq[:,hist_len:pred_keytime[0][-1]+1,-self.action_dim:] # do not support batch evaluation
            #         pred_action = pred_action_seq[:,hist_len:,:] # do not support batch evaluation
            #     else:
            #     #     # print("no keyframe", timesteps[0,-1,0], pred_keytime_differences, self.normalizer.unnormalize(pred_keyframe))
            #         # pred_action = pred_action_seq[:,hist_len:hist_len+self.eval_action_len,-self.action_dim:] # do not support batch evaluation
            #         pred_action = pred_action_seq[:,hist_len:hist_len+self.n_obs_steps,-self.action_dim:] # do not support batch evaluation
            # # Only used for ms-skill challenge online evaluation
            # # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),:]
            # # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            # #     for i in range(self.eval_action_len-1):
            # #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,:])
        
        return pred_action
    
    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            if False: # self.train_keyframe_model and (not self.train_diff_model):
                self.normalizer = None
            else:
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
            sampled_batch = memory.sample(batch_size, device=self.device, obs_mask=self.obs_mask, require_mask=True, obsact_normalizer=self.normalizer, keyframe_type=self.keyframe_model_type)
        else:
            sampled_batch = memory.sample(batch_size, device=self.device, obs_mask=self.obs_mask, require_mask=True, action_normalizer=self.normalizer, keyframe_type=self.keyframe_model_type)
        # sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True) # ["obs","actions"] # Did in replay buffer
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.keyframe_lr_scheduler is not None:
            self.keyframe_lr_scheduler.step()
        
        if self.train_diff_model and self.actor_optim is not None:
            self.actor_optim.zero_grad()
        if self.train_keyframe_model and self.keyframe_optim is not None:
            self.keyframe_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)], 
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1), 
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        loss = 0.
        ret_dict = {}

        ep_first_obs = None
        if self.use_ep_first_obs and ('ep_first_obs' in sampled_batch):
            ep_first_obs = sampled_batch['ep_first_obs']
            # for key in ep_first_obs:
            #     print(ep_first_obs[key].shape)
        
        # generate impainting mask
        if self.diffuse_state:
            if self.pose_only:
                if self.extra_dim > 0:
                    traj_data = torch.cat([sampled_batch["normed_states"][...,-self.pose_dim-self.extra_dim:-self.extra_dim],sampled_batch["normed_actions"]], dim=-1)  # We only preserve the tcp pose for diffusion
                else:
                    traj_data = torch.cat([sampled_batch["normed_states"][...,-self.pose_dim:],sampled_batch["normed_actions"]], dim=-1)  # We only preserve the tcp pose for diffusion
            else:
                traj_data = torch.cat([sampled_batch["normed_states"],sampled_batch["normed_actions"]], dim=-1)
        else:
            traj_data = sampled_batch["normed_actions"]
        # Need Normalize! (Already did in replay buffer)
        # traj_data = self.normalizer.normalize(traj_data)
        masked_obs = sampled_batch['obs'] # This is not normalized
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
                raise NotImplementedError("Not support obs not as cond! Please set obs_as_global_cond=True")

        if self.train_diff_model:
            if (self.diffusion_updates is None) or ((self.diffusion_updates is not None) and updates <= self.diffusion_updates):
                obs_fea = self.obs_encoder(masked_obs, ep_first_obs_dict=ep_first_obs)

                diff_loss, info = self.diff_loss(x=traj_data, masks=sampled_batch["is_valid"], cond_mask=data_mask, global_cond=obs_fea) # TODO: local_cond, returns
                ret_dict.update(info)
                loss += diff_loss

            else:
                self.train_diff_model = False
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.obs_encoder.parameters():
                    param.requires_grad = False

        if self.train_keyframe_model:
            if (self.keyframe_model_updates is None) or ((self.keyframe_model_updates is not None) and updates <= self.keyframe_model_updates):
                keyframe_actions = sampled_batch["keyframe_actions"]
                keyframe_states = sampled_batch["keyframe_states"]
                # keyframes = self.normalizer.normalize(keyframes)
                keytime_differences = sampled_batch["keytime_differences"]
                keyframe_masks = sampled_batch["keyframe_masks"]

                if self.keyframe_model_type == "bc":
                    keyframe_obs_fea = self.keyframe_obs_encoder(masked_obs, ep_first_obs_dict=ep_first_obs)

                    keyframe_states = keyframe_states[:,obs_mask,...][:,-1] # We only take the last step of the horizon since we want to train the key frame model
                    keyframe_masks = keyframe_masks[:,obs_mask,...][:,-1]
                    keytime_differences = keytime_differences[:,obs_mask,...][:,-1]
                    keyframe_loss, info = self.keyframe_bc_loss(keyframe_obs_fea, keyframe_states, keytime_differences, keyframe_masks)
                
                elif self.keyframe_model_type == "gpt":
                    timesteps = sampled_batch["timesteps"]
                    keyframe_obs_fea = states = masked_obs["state"]
                    if not self.keyframe_state_only:
                        img_obs_fea = self.keyframe_obs_encoder(masked_obs, ep_first_obs_dict=ep_first_obs, img_fea_only=True)
                        keyframe_obs_fea = torch.cat([img_obs_fea, states], dim=-1)
                    ep_first_state = None
                    if ep_first_obs is not None:
                        if len(ep_first_obs['state'].shape) == 2:
                            ep_first_obs['state'] = ep_first_obs['state'].unsqueeze(1)
                         # Append ep first obs for predicting keyframes
                        ep_first_state = ep_first_obs['state']
                    actions = sampled_batch["actions"][:,obs_mask,...]
                    keyframe_states = keyframe_states[:,obs_mask,...][:,-1] # We only take the last step of the horizon since we want to train the key frame model
                    keyframe_actions = keyframe_actions[:,obs_mask,...][:,-1]
                    keytime_differences = keytime_differences[:,obs_mask,...][:,-1]
                    keyframe_masks = keyframe_masks[:,obs_mask,...][:,-1]

                    keyframe_loss, info = self.keyframe_gpt_loss(keyframe_obs_fea, timesteps, actions, keyframe_states, keyframe_actions, keytime_differences, keyframe_masks, ep_first_state=ep_first_state)
                
                ret_dict.update(info)
                loss += keyframe_loss
        
        loss.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), 1.0) # This may cause diffusion not work!!!!
        # for param in self.keyframe_obs_encoder.parameters():
        #     print(param.name, param.grad)
        if self.train_diff_model and (self.actor_optim is not None):
            self.actor_optim.step()
        if self.train_keyframe_model and (self.keyframe_optim is not None):
            self.keyframe_optim.step()

        ## Not implement yet
        # if self.step % self.update_ema_every == 0:
        #     self.step_ema()
        if self.train_diff_model:
            ret_dict["grad_norm_diff_model"] = np.mean([torch.linalg.norm(parameter.grad.data).item() for parameter in self.model.parameters() if parameter.grad is not None])
            ret_dict["grad_norm_diff_obs_encoder"] = np.mean([torch.linalg.norm(parameter.grad.data).item() for parameter in self.obs_encoder.parameters() if parameter.grad is not None])

        if self.lr_scheduler is not None:
            ret_dict["lr"] = get_mean_lr(self.actor_optim)
        ret_dict = dict(ret_dict)
        ret_dict = {'diffusion/' + key: val for key, val in ret_dict.items()}
        
        self.step += 1

        return ret_dict