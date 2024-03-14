"""
Diffusion Policy
"""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from maniskill2_learn.networks import build_model
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.torch import BaseAgent, get_mean_lr, build_optimizer
from maniskill2_learn.utils.diffusion.helpers import Losses
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.diffusion.progress import Progress, Silent
from maniskill2_learn.utils.diffusion.mask_generator import LowdimMaskGenerator
from maniskill2_learn.utils.diffusion.normalizer import LinearNormalizer
from maniskill2_learn.utils.diffusion.dict_of_mixin import DictOfTensorMixin

from ..builder import BRL


@BRL.register_module()
class DiffAgent(BaseAgent):
    def __init__(
        self,
        nn_cfg: dict,
        optim_cfg: dict,
        env_params: dict,
        action_seq_len: int,
        eval_action_len: int = 1,
        visual_nn_cfg: dict = None,
        pcd_cfg: Optional[dict] = None,
        lr_scheduler_cfg: Optional[dict] = None,
        batch_size: int = 256,
        n_timesteps: int = 150,
        loss_type: str = "l1",
        clip_denoised: bool = False,
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights: Optional[np.ndarray] = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 0.1,
        agent_share_noise: bool = False,
        obs_as_global_cond: bool = True,  # diffuse action or take obs as condition inputs
        action_visible: bool = True,  # If we cond on some hist actions
        fix_obs_steps: bool = True,  # Randomly cond on certain obs steps or deterministicly
        n_obs_steps: int = 3,
        normalizer: DictOfTensorMixin = LinearNormalizer(),
        diffuse_state: bool = False,
        pose_only: bool = False,
        pose_dim: int = 7,
        extra_dim: int = 0,
        **kwargs,
    ):
        super(DiffAgent, self).__init__()
        self.batch_size = batch_size

        if visual_nn_cfg is not None:
            if pcd_cfg is not None:
                visual_nn_cfg["pcd_model"] = build_model(pcd_cfg)
            visual_nn_cfg["n_obs_steps"] = n_obs_steps
            self.obs_encoder = build_model(visual_nn_cfg)
            self.img_feature_dim = self.obs_encoder.img_feature_dim
            self.obs_feature_dim = self.obs_encoder.out_feature_dim
        else:
            self.obs_encoder = None
            self.obs_feature_dim = None

        lr_scheduler_cfg = lr_scheduler_cfg
        self.action_dim = env_params["action_shape"]
        self.diffuse_state = diffuse_state

        if self.obs_feature_dim is not None:
            nn_cfg.update(dict(global_cond_dim=self.obs_feature_dim))
        else:
            nn_cfg.update(dict(global_cond_dim=nn_cfg["global_cond_dim"] * n_obs_steps))
        self.model = build_model(nn_cfg)
        self.normalizer = normalizer

        self.horizon = self.action_seq_len = action_seq_len
        self.observation_shape = env_params["obs_shape"]

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.agent_share_noise = agent_share_noise

        self.step = 0

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.noise_scheduler = DDPMScheduler(
            beta_schedule="squaredcos_cap_v2",
            clip_sample=self.clip_denoised,
            num_train_timesteps=self.n_timesteps,
        )

        self.loss_type = loss_type
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        if self.obs_encoder is not None:
            self.actor_optim = build_optimizer(
                [self.model, self.obs_encoder], optim_cfg
            )
        else:
            self.actor_optim = build_optimizer(self.model, optim_cfg)

        if lr_scheduler_cfg is None:
            self.lr_scheduler = None
        else:
            lr_scheduler_cfg["optimizer"] = self.actor_optim
            self.lr_scheduler = build_lr_scheduler(lr_scheduler_cfg)

        self.extra_parameters = dict(kwargs)

        obs_dim = 0
        if not obs_as_global_cond:
            obs_dim = self.obs_feature_dim
        if diffuse_state:
            # self.state_dim = env_params["obs_shape"]["state"]
            obs_dim = env_params["obs_shape"]["state"]
            if pose_only:
                # obs_dim = pose_dim # We only diffuse tcp pose
                obs_dim = (
                    pose_dim + extra_dim
                )  # We only diffuse tcp pose and the target pose

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=fix_obs_steps,
            action_visible=action_visible,
            return_one_mask=diffuse_state,
        )
        self.obs_as_global_cond = obs_as_global_cond
        self.action_visible = action_visible
        self.fix_obs_steps = fix_obs_steps
        self.n_obs_steps = n_obs_steps

        self.init_normalizer = False
        self.pose_only = pose_only

        self.act_mask, self.obs_mask, self.data_mask = None, None, None

        self.eval_action_len = eval_action_len

        self.pose_dim = pose_dim
        self.extra_dim = extra_dim

        # Only used for ms-skill challenge online evaluation
        # self.eval_action_queue = None
        # if self.eval_action_len > 1:
        #     self.eval_action_queue = deque(maxlen=self.eval_action_len-1)

    def eval(self):
        return super().eval()

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory
        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).clone()

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond_data,
        cond_mask,
        local_cond=None,
        global_cond=None,
        returns=None,
        verbose=True,
        return_diffusion=False,
        **kwargs,
    ):
        x = torch.randn(
            size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device
        )

        if return_diffusion:
            diffusion = [x]

        n_timesteps = len(self.noise_scheduler.timesteps)
        progress = Progress(n_timesteps) if verbose else Silent()
        for i, t in enumerate(self.noise_scheduler.timesteps):
            # 1. apply conditioning
            x[cond_mask] = cond_data[cond_mask]

            # 2. predict model output
            if self.returns_condition:
                epsilon_cond = self.model(
                    x, t, local_cond, global_cond, returns, use_dropout=False
                )
                epsilon_uncond = self.model(
                    x, t, local_cond, global_cond, returns, force_dropout=True
                )
                epsilon = epsilon_uncond + self.condition_guidance_w * (
                    epsilon_cond - epsilon_uncond
                )
            else:
                epsilon = self.model(x, t, local_cond, global_cond)

            # 3. compute previous image: x_t -> x_t-1
            x = self.noise_scheduler.step(epsilon, t, x).prev_sample

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()
        # 4. finally make sure conditioning is enforced
        x[cond_mask] = cond_data[cond_mask]

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self,
        cond_data,
        cond_mask,
        local_cond=None,
        global_cond=None,
        returns=None,
        *args,
        **kwargs,
    ):
        """
        conditions : [ (time, state), ... ]
        """

        # horizon = action_seq_len or self.action_seq_len
        # batch_size = len(list(cond_data.values())[0])
        # shape = (batch_size, horizon, self.action_dim) # cond_data.shape
        return self.p_sample_loop(
            cond_data, cond_mask, local_cond, global_cond, returns, *args, **kwargs
        )

    # ------------------------------------------ training ------------------------------------------#

    def p_losses(
        self, actions, t, cond_mask, local_cond=None, global_cond=None, returns=None
    ):
        noise = torch.randn_like(actions)
        action_noisy = self.noise_scheduler.add_noise(actions, noise, t)
        # apply conditioning
        action_noisy[cond_mask] = actions[cond_mask]

        pred = self.model(action_noisy, t, local_cond, global_cond, returns)
        assert noise.shape == pred.shape

        loss = F.mse_loss(pred[~cond_mask], noise[~cond_mask], reduction="none")
        return loss, {}

    def loss(
        self, x, masks, cond_mask, local_cond=None, global_cond=None, returns=None
    ):
        # x is the action, with shape (bs, horizon, act_dim)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(
            x, t, cond_mask, local_cond, global_cond, returns
        )
        # diffuse_loss = (diffuse_loss * masks.unsqueeze(-1)).mean()
        diffuse_loss = diffuse_loss.mean()
        info.update({"action_diff_loss": diffuse_loss.item()})
        return diffuse_loss, info

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

        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
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
            act_mask = act_mask[: action_history.shape[0]]  # obs mask is int

        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:, obs_mask, ...]

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

        if mode == "eval":
            pred_action = pred_action[:, -(self.action_seq_len - hist_len):]
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
                sampled_batch["normed_states"] = sampled_batch["normed_states"][:, self.obs_mask]
            masked_obs["state"] = sampled_batch["normed_states"]
        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask, _ = self.mask_generator(
                    traj_data.shape, self.device
                )
                self.act_mask, self.obs_mask = act_mask, obs_mask
                for key in masked_obs:
                    masked_obs[key] = masked_obs[key][:, obs_mask, ...]
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
            cond_mask=act_mask,
            global_cond=obs_fea,
        )  # TODO: local_cond, returns
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

    # Not implement yet
    # def step_ema(self):
    #     if self.step < self.step_start_ema:
    #         self.reset_parameters()
    #         return
    #     self.ema.update_model_average(self.ema_model, self.model)
