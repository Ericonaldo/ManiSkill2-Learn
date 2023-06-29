"""
Diffusion Policy
"""
import numpy as np
from copy import copy
import random
import torch
import torch.nn.functional as F

from maniskill2_learn.networks import build_model, build_reg_head
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.data import DictArray, GDict, dict_to_str
from maniskill2_learn.utils.meta import get_total_memory, get_logger
from maniskill2_learn.utils.torch import (
    BaseAgent,
    get_mean_lr,
    get_cuda_info,
    build_optimizer,
)
from maniskill2_learn.utils.diffusion.jump_diff_utils import get_forward_rate
from maniskill2_learn.utils.diffusion.helpers import (
    Losses,
    apply_conditioning,
    cosine_beta_schedule,
    extract,
)
from maniskill2_learn.utils.diffusion.arrays import to_torch, to_np
from maniskill2_learn.utils.diffusion.progress import Progress, Silent
from maniskill2_learn.utils.diffusion.mask_generator import LowdimMaskGenerator
from maniskill2_learn.utils.diffusion.normalizer import LinearNormalizer

# from maniskill2_learn.networks.modules.cnn_modules.multi_image_obs_encoder import MultiImageObsEncoder

from ..builder import BRL


@BRL.register_module()
class JumpDiffAgent(BaseAgent):
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
        obs_as_global_cond=True,  # diffuse action or take obs as condition inputs
        action_visible=True,  # If we cond on some hist actions
        fix_obs_steps=True,  # Randomly cond on certain obs steps or deterministicly
        n_obs_steps=3,
        normalizer=LinearNormalizer(),
        jump_rate_function_name="step",
        jump_rate_cur_t=0.1,
        **kwargs,
    ):
        super(JumpDiffAgent, self).__init__()
        self.batch_size = batch_size

        if pcd_cfg is not None:
            visual_nn_cfg["pcd_model"] = build_model(pcd_cfg)
        visual_nn_cfg["n_obs_steps"] = n_obs_steps
        self.obs_encoder = build_model(visual_nn_cfg)
        self.obs_feature_dim = self.obs_encoder.out_feature_dim

        lr_scheduler_cfg = lr_scheduler_cfg
        self.action_dim = env_params["action_shape"]
        self.normalizer = normalizer

        actor_cfg["action_seq_len"] = action_seq_len
        actor_cfg.update(env_params)
        self.actor = build_model(actor_cfg)
        nn_cfg.update(dict(global_cond_dim=self.obs_feature_dim))
        self.model = build_model(nn_cfg)

        self.horizon = self.action_seq_len = action_seq_len
        self.observation_shape = env_params["obs_shape"]

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.agent_share_noise = agent_share_noise

        self.step = 0

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_type = loss_type
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        self.actor_optim = build_optimizer([self.model, self.obs_encoder], optim_cfg)
        if lr_scheduler_cfg is None:
            self.lr_scheduler = None
        else:
            lr_scheduler_cfg["optimizer"] = self.actor_optim
            self.lr_scheduler = build_lr_scheduler(lr_scheduler_cfg)

        self.extra_parameters = dict(kwargs)

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if obs_as_global_cond else self.obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=fix_obs_steps,
            action_visible=action_visible,
        )
        self.obs_as_global_cond = obs_as_global_cond
        self.action_visible = action_visible
        self.fix_obs_steps = fix_obs_steps
        self.n_obs_steps = n_obs_steps
        self.n_future_steps = self.action_seq_len - self.n_obs_steps

        self.jump_forward_rate = get_forward_rate(
            jump_rate_function_name,
            self.n_future_steps,
            jump_rate_cur_t,
        )

        self.init_normalizer = False

        self.act_mask, self.obs_mask = None, None

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

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, local_cond=None, global_cond=None, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond, *_ = self.model(
                x, t, local_cond, global_cond, returns, use_dropout=False
            )
            epsilon_uncond, *_ = self.model(
                x, t, local_cond, global_cond, returns, force_dropout=True
            )
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon, *_ = self.model(x, t, local_cond, global_cond)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, n_future_steps_xt, local_cond=None, global_cond=None, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, local_cond=local_cond, global_cond=global_cond, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
            n_future_steps_xt,
        )

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
    ):
        device = self.betas.device

        batch_size = cond_data.shape[0]
        x = torch.randn(
            size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device
        )
        n_future_steps_xt = torch.ones(batch_size, dtype=torch.long, device=device)
        x = self.delete_steps(x, n_future_steps_xt * self.n_future_steps, n_future_steps_xt)

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            # 1. apply conditioning
            x[cond_mask] = cond_data[cond_mask]

            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # 2. predict model output and replace sample
            x, n_future_steps_xt = self.p_sample(x, timesteps, n_future_steps_xt, local_cond, global_cond, returns)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

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
        action_seq_len=None,
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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def delete_steps(self, actions, n_future_steps, new_n_future_steps, return_deleted=False):
        if return_deleted:
            deleted_actions, deleted_index = [], []

        batch_size = len(actions)
        new_actions = torch.zeros_like(actions)
        new_actions[:, :self.n_obs_steps] = actions[:, :self.n_obs_steps]
        for idx in range(batch_size):
            if new_n_future_steps[idx] == n_future_steps[idx]:
                new_actions[idx][self.n_obs_steps:] = actions[idx][self.n_obs_steps:]
                if return_deleted:
                    deleted_actions.append([])
                    deleted_index.append(None)
                continue

            remain_pred_steps = torch.tensor(random.sample(range(n_future_steps[idx]), new_n_future_steps[idx]))
            new_actions[idx][self.n_obs_steps: self.n_obs_steps + new_n_future_steps[idx]] = actions[idx][remain_pred_steps + self.n_obs_steps]
            new_actions[idx][self.n_obs_steps + new_n_future_steps[idx]:] = new_actions[idx][self.n_obs_steps + new_n_future_steps[idx] - 1]
            if return_deleted:
                delete_pred_steps = torch.tensor(list(set(range(n_future_steps[idx])) - set(to_np(remain_pred_steps).tolist())), device=self.device) + self.n_obs_steps
                deleted_actions.append(actions[idx][delete_pred_steps])
                deleted_index.append(delete_pred_steps)
        if return_deleted:
            return new_actions, deleted_actions, deleted_index
        else:
            return new_actions

    def delete_one_step(self, actions, n_future_steps):
        new_n_future_steps = torch.clamp(n_future_steps - 1, min=1)
        del_one_actions, deleted_actions, deleted_index = self.delete_steps(actions, n_future_steps, new_n_future_steps, return_deleted=True)
        deleted_actions = torch.cat(
            [act if len(act) > 0 else torch.zeros_like(actions[0, :1]) for act in deleted_actions], dim=0
        )
        deleted_index = torch.cat(
            [idx if idx is not None else torch.zeros_like(n_future_steps[:1]) for idx in deleted_index], dim=0
        )
        return del_one_actions, new_n_future_steps, deleted_actions, deleted_index

    def p_losses(
        self, actions, t, cond_mask, x0_n_future_steps, local_cond=None, global_cond=None, returns=None
    ):
        n_future_steps_xt = self.jump_forward_rate.get_dims_at_t(
            start_dims=x0_n_future_steps, ts=t / self.n_timesteps,
        ).int().to(self.device)
        actions = self.delete_steps(actions, x0_n_future_steps, new_n_future_steps=n_future_steps_xt)

        noise = torch.randn_like(actions)
        noise = self.delete_steps(noise, x0_n_future_steps, new_n_future_steps=n_future_steps_xt)
        action_noisy = self.q_sample(x_start=actions, t=t, noise=noise)
        # apply conditioning
        action_noisy[cond_mask] = actions[cond_mask]

        # first network pass
        pred, _, x0_step_logits, _ = self.model(action_noisy, t, local_cond, global_cond, returns)
        assert noise.shape == pred.shape, f"{noise.shape} != {pred.shape}"

        # second network pass
        del_one_action_noisy = copy(action_noisy)
        del_one_action_noisy, del_one_n_future_steps, deleted_action_noisy, deleted_index = self.delete_one_step(del_one_action_noisy, n_future_steps_xt)
        _, index_logits, _, pred_add_value = self.model(action_noisy, t, local_cond, global_cond, returns)

        # f_rate_vs_t = self.jump_forward_rate.get_rate(n_future_steps_xt, ts=t / self.n_timesteps).to(self.device)  # (B,)
        add_loss_masks = (del_one_n_future_steps != n_future_steps_xt).unsqueeze(-1)
        # pred_add_loss = F.mse_loss(pred_add_value, deleted_action_noisy, reduction="none") * f_rate_vs_t.unsqueeze(-1)
        pred_add_loss = F.mse_loss(pred_add_value, deleted_action_noisy, reduction="none")
        pred_add_loss = (pred_add_loss * add_loss_masks).mean(dim=-1).sum() / add_loss_masks.sum()

        x0_step_ce_loss = F.cross_entropy(x0_step_logits, x0_n_future_steps - 1)

        target_index = torch.clamp(deleted_index - self.n_obs_steps, min=0)
        index_loss_masks = add_loss_masks.squeeze(-1)
        # index_loss = F.cross_entropy(index_logits, target_index, reduction="none") * f_rate_vs_t
        index_loss = F.cross_entropy(index_logits, target_index, reduction="none")
        index_loss = (index_loss * index_loss_masks).sum() / index_loss_masks.sum()

        pred_masks = torch.ones_like(pred)
        pred_masks[:, : self.n_obs_steps - 1] = 0.0  # '-1' since we need to diffuse actions for current step
        for idx in range(len(pred_masks)):
            pred_masks[idx][self.n_obs_steps + n_future_steps_xt[idx]:] = 0.0
        if self.predict_epsilon:
            diff_loss = F.mse_loss(pred, noise, reduction="none")
        else:
            diff_loss = F.mse_loss(pred, actions, reduction="none")
        diff_loss = (diff_loss * pred_masks).sum() / pred_masks.sum()

        # loss = diff_loss + (pred_add_loss + x0_step_ce_loss + index_loss) / actions.shape[1]
        loss = diff_loss + pred_add_loss + x0_step_ce_loss + index_loss

        return loss, {
            "action_diff_loss": diff_loss.detach().cpu(),
            "pred_add_loss": pred_add_loss.detach().cpu(),
            "x0_step_ce_loss": x0_step_ce_loss.detach().cpu(),
            "index_loss": index_loss.detach().cpu(),
        }

    def loss(
        self, x, masks, cond_mask, local_cond=None, global_cond=None, returns=None
    ):
        # x is the action, with shape (bs, horizon, act_dim)
        batch_size = len(x)
        # pred_len = (cond_mask[..., 0] == False).sum(dim=-1)
        # history_len = x.shape[1] - pred_len
        # XXX(zbzhu): check this
        cropped_n_pred_steps = (torch.rand(batch_size, device=self.device) * (self.n_future_steps - 1)).to(int) + 1
        cropped_x = copy(x)
        for idx in range(len(cropped_x)):
            remain_len = cropped_n_pred_steps[idx] + self.n_obs_steps
            cropped_x[idx][remain_len:] = cropped_x[idx][remain_len - 1]

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(
            cropped_x, t, cond_mask, cropped_n_pred_steps, local_cond, global_cond, returns
        )
        # diffuse_loss = (diffuse_loss * masks.unsqueeze(-1)).mean()
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
                act_mask, obs_mask = self.mask_generator(
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

        obs_fea = self.obs_encoder(
            observation
        )  # No need to mask out since the history is set as the desired length

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
        pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
        pred_action = pred_action_seq

        if mode == "eval":
            pred_action = pred_action_seq[:, -(self.action_seq_len - hist_len) :, :]
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
            self.normalizer.fit(data, last_n_dims=1, mode="limits")
            self.init_normalizer = True

        batch_size = self.batch_size
        sampled_batch = memory.sample(
            batch_size,
            device=self.device,
            obs_mask=self.obs_mask,
            require_mask=True,
            action_normalizer=self.normalizer,
        )
        # sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True) # ["obs","actions"] # Did in replay buffer

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.actor_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)],
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1),
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        # generate impainting mask
        traj_data = sampled_batch[
            "actions"
        ]  # Need Normalize! (Already did in replay buffer)
        # traj_data = self.normalizer.normalize(traj_data)
        masked_obs = sampled_batch["obs"]
        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask = self.mask_generator(traj_data.shape, self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
                for key in masked_obs:
                    # HACK(zbzhu)
                    if isinstance(masked_obs[key], list):
                        masked_obs[key] = masked_obs[key][0][:, obs_mask, ...]
                    else:
                        masked_obs[key] = masked_obs[key][:, obs_mask, ...]
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        obs_fea = self.obs_encoder(masked_obs)

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
