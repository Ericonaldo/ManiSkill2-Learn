"""
Diffusion Policy
"""

import numpy as np
import torch
from transforms3d.quaternions import quat2axangle

from maniskill2_learn.methods.brl import DiffAgent
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.torch import get_mean_lr
from maniskill2_learn.utils.diffusion.mask_generator import KeyframeMaskGenerator

from ..builder import BRL


def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta


@BRL.register_module()
class KeyframeDiffAgent(DiffAgent):
    def __init__(
        self,
        mask_generator_cls: callable = KeyframeMaskGenerator,
        control_mode: str = "pd_ee_pose",
        rot_rep: str = "quat",
        *args,
        **kwargs,
    ):
        assert mask_generator_cls == KeyframeMaskGenerator, mask_generator_cls
        self.control_mode = control_mode
        self.rot_rep = rot_rep
        super().__init__(mask_generator_cls=mask_generator_cls, *args, **kwargs)

    def forward(
        self,
        observation: np.ndarray,
        next_keyframe: np.ndarray,
        mode: str = "eval",
        grip_thresh: float = 0.03,
        *args,
        **kwargs,
    ):
        # if mode == "eval":  # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()

        if self.control_mode == "pd_joint_pos":
            keyframe = np.concatenate(
                observation["state"][:, :7],
                np.where(
                    observation["state"][:, 8:9] > grip_thresh, 1.0, -1.0
                ),  # gripper open
                axis=-1,
            )
        elif self.control_mode == "pd_ee_pose":
            if self.rot_rep == "quat":
                keyframe = np.concatenate(
                    [
                        observation["state"][:, 18 : 18 + 3],  # ee pos
                        # the maniskill pose controller take axis angle as action input
                        np.array(list(map(
                            compact_axis_angle_from_quaternion,
                            observation["state"][:, 18 + 3 : 18 + 7],
                        ))),  # ee pose [pos, axis-angle]
                        np.where(
                            observation["state"][:, 8:9] > grip_thresh, 1.0, -1.0
                        ),  # gripper open
                    ],
                    axis=-1,
                )
            else:
                raise NotImplementedError(f"Unknown rotation representation: {self.rot_rep}")
        else:
            raise NotImplementedError(f"Unknown control mode: {self.control_mode}")

        keyframe = np.stack([keyframe, next_keyframe], axis=1)

        observation = to_torch(observation, device=self.device, dtype=torch.float32)
        keyframe_action = to_torch(keyframe, device=self.device, dtype=torch.float32)
        action_dim = keyframe_action.shape[-1]

        if self.obs_encoder is None:
            # Pad and normalize state
            data = torch.cat(
                [
                    observation["state"],
                    torch.zeros(
                        [*observation["state"].shape[:-1], action_dim],
                        device=self.device,
                        dtype=torch.float32,
                    )
                ],
                dim=-1,
            )
            data = self.normalizer.normalize(data)
            observation["state"] = data[..., : observation["state"].shape[-1]]

            # Pad and normalize action
            data = torch.cat(
                [
                    torch.zeros(
                        [*keyframe_action.shape[:-1], observation["state"].shape[-1]],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    keyframe_action,
                ],
                dim=-1,
            )
            data = self.normalizer.normalize(data)
            keyframe_action = data[..., -self.action_dim :]
        else:
            keyframe_action = self.normalizer.normalize(keyframe_action)

        bs = keyframe_action.shape[0]
        self.set_mode(mode=mode)

        act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask, _ = self.mask_generator(
                    (bs, self.horizon, self.action_dim), self.device
                )
                self.act_mask, self.obs_mask = act_mask, obs_mask
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        if act_mask.shape[0] < bs:
            act_mask = act_mask.repeat(max(bs // act_mask.shape[0] + 1, 2), 1, 1)
        if act_mask.shape[0] != bs:
            act_mask = act_mask[: keyframe_action.shape[0]]  # obs mask is int

        if self.obs_encoder is not None:
            obs_fea = self.obs_encoder(
                observation
            )  # No need to mask out since the history is set as the desired length
        else:
            obs_fea = observation["state"].reshape(bs, -1)

        supp = torch.zeros(
            bs,
            self.horizon - 2,
            self.action_dim,
            dtype=keyframe_action.dtype,
            device=self.device,
        )
        cond_action_sequence = torch.cat(
            [keyframe_action[:, 0:1], supp, keyframe_action[:, -1:]], dim=1
        )
        pred_action_seq = self.conditional_sample(
            cond_data=cond_action_sequence,
            cond_mask=act_mask,
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
