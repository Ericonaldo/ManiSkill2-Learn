"""
Diffusion Policy
"""

import numpy as np
import torch

from maniskill2_learn.methods.brl import DiffAgent
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.torch import get_mean_lr
from maniskill2_learn.utils.diffusion.mask_generator import KeyframeMaskGenerator

from ..builder import BRL


@BRL.register_module()
class KeyframeDiffAgent(DiffAgent):
    def __init__(self, mask_generator_cls: callable = KeyframeMaskGenerator, *args, **kwargs):
        assert mask_generator_cls == KeyframeMaskGenerator, mask_generator_cls
        super().__init__(mask_generator_cls=mask_generator_cls, *args, **kwargs)

    def forward(
        self,
        observation: np.ndarray,
        returns_rate: float = 0.9,
        mode: str = "eval",
        *args,
        **kwargs,
    ):
        # if mode == "eval":  # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()

        observation = to_torch(observation, device=self.device, dtype=torch.float32)

        action_history = observation["actions"]
        if self.obs_encoder is None:
            action_history = torch.cat(
                [action_history, torch.zeros_like(action_history[:, :1])],
                dim=1,
            )
            data = self.normalizer.normalize(
                torch.cat((observation["state"], action_history), dim=-1)
            )
            observation["state"] = data[..., : observation["state"].shape[-1]]
            action_history = data[:, :-1, -self.action_dim :]
        else:
            action_history = self.normalizer.normalize(action_history)
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")

        self.set_mode(mode=mode)

        if self.act_mask is None or self.obs_mask is None:
            if self.obs_as_global_cond:
                self.act_mask, self.obs_mask, _ = self.mask_generator(
                    (bs, self.horizon, self.action_dim), self.device
                )
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        if self.act_mask.shape[0] < bs:
            self.act_mask = self.act_mask.repeat(max(bs // self.act_mask.shape[0] + 1, 2), 1, 1)
        if self.act_mask.shape[0] != bs:
            self.act_mask = self.act_mask[: action_history.shape[0]]  # obs mask is int

        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:, self.obs_mask, ...]

        if self.obs_encoder is not None:
            obs_fea = self.obs_encoder(
                observation
            )  # No need to mask out since the history is set as the desired length
        else:
            obs_fea = observation["state"].reshape(bs, -1)

        if self.action_seq_len - hist_len:
            supp = torch.zeros(
                bs,
                self.action_seq_len - hist_len,
                self.action_dim,
                dtype=action_history.dtype,
                device=self.device,
            )
            action_history = torch.concat([action_history, supp], dim=1)

        pred_action_seq = self.conditional_sample(
            cond_data=action_history,
            cond_mask=self.act_mask,
            global_cond=obs_fea,
            *args,
            **kwargs,
        )
        data = pred_action_seq
        if self.obs_encoder is None:
            supp = torch.zeros(
                *pred_action_seq.shape[:-1],
                observation["state"].shape[-1],
                dtype=pred_action_seq.dtype,
                device=self.device,
            )
            data = torch.cat([supp, pred_action_seq], dim=-1)
        data = self.normalizer.unnormalize(data)
        pred_action = data[..., -self.action_dim :]

        if mode == "eval":
            pred_action = pred_action[:, -(self.action_seq_len - hist_len) :]
            # Only used for ms-skill challenge online evaluation
            # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),-self.action_dim:]
            # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            #     for i in range(self.eval_action_len-1):
            #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,-self.action_dim:])

        return pred_action

    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            if self.obs_encoder is None:
                data = np.concatenate(
                    (memory.get_all("obs", "state"), memory.get_all("actions")), axis=-1
                )
            else:
                data = memory.get_all("actions")
            self.normalizer.fit(data, last_n_dims=1, mode="limits", range_eps=1e-7)
            self.init_normalizer = True

        batch_size = self.batch_size
        if self.obs_encoder is None:
            sample_kwargs = {"obsact_normalizer": self.normalizer}
        else:
            sample_kwargs = {"action_normalizer": self.normalizer}
        sampled_batch = memory.sample(
            batch_size,
            device=self.device,
            obs_mask=self.obs_mask,
            require_mask=True,
            **sample_kwargs,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.actor_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)],
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1),
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        # generate impainting mask
        traj_data = sampled_batch["normed_actions"]
        masked_obs = sampled_batch["obs"]
        if self.obs_encoder is None:
            if self.obs_mask is not None:
                sampled_batch["normed_states"] = sampled_batch["normed_states"][
                    :, self.obs_mask
                ]
            masked_obs["state"] = sampled_batch["normed_states"]

        if self.act_mask is None or self.obs_mask is None:
            if self.obs_as_global_cond:
                self.act_mask, self.obs_mask, _ = self.mask_generator(
                    traj_data.shape, self.device
                )
                for key in masked_obs:
                    masked_obs[key] = masked_obs[key][:, self.obs_mask, ...]
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        if self.obs_encoder is not None:
            obs_fea = self.obs_encoder(masked_obs)
        else:
            obs_fea = masked_obs["state"].reshape(batch_size, -1)

        loss, ret_dict = self.loss(
            x=traj_data,
            masks=sampled_batch["is_valid"],
            cond_mask=self.act_mask,
            global_cond=obs_fea,
        )
        loss.backward()
        self.actor_optim.step()

        # Not implement yet
        # if self.step % self.update_ema_every == 0:
        #     self.step_ema()

        ret_dict["grad_norm_diff_model"] = np.mean(
            [
                torch.linalg.norm(parameter.grad.data).item()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
        )
        if self.obs_encoder is not None:
            ret_dict["grad_norm_diff_obs_encoder"] = np.mean(
                [
                    torch.linalg.norm(parameter.grad.data).item()
                    for parameter in self.obs_encoder.parameters()
                    if parameter.grad is not None
                ]
            )

        if self.lr_scheduler is not None:
            ret_dict["lr"] = get_mean_lr(self.actor_optim)
        ret_dict = dict(ret_dict)
        ret_dict = {"diffusion/" + key: val for key, val in ret_dict.items()}

        self.step += 1

        return ret_dict
