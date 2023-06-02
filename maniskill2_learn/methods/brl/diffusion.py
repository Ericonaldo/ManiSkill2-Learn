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
from collections import defaultdict, deque

from maniskill2_learn.networks import build_model, build_reg_head
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.data import to_torch, DictArray, GDict, dict_to_str
from maniskill2_learn.utils.meta import get_total_memory, get_logger
from maniskill2_learn.utils.torch import BaseAgent, get_mean_lr, get_cuda_info, build_optimizer
from maniskill2_learn.utils.diffusion.helpers import Losses, apply_conditioning, cosine_beta_schedule, extract
from maniskill2_learn.utils.diffusion.progress import Progress, Silent
from maniskill2_learn.utils.diffusion.mask_generator import LowdimMaskGenerator
from maniskill2_learn.utils.diffusion.normalizer import LinearNormalizer
# from maniskill2_learn.networks.modules.cnn_modules.multi_image_obs_encoder import MultiImageObsEncoder

from ..builder import BRL


@BRL.register_module()
class DiffAgent(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        visual_nn_cfg,
        nn_cfg,
        optim_cfg,
        env_params,
        action_seq_len,
        lr_scheduler_cfg=None,
        batch_size=128,
        n_timesteps=1000,
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
        super(DiffAgent, self).__init__()
        self.batch_size = batch_size

        visual_nn_cfg['n_obs_steps'] = n_obs_steps
        self.obs_encoder = build_model(visual_nn_cfg)
        self.obs_feature_dim = self.obs_encoder.out_feature_dim

        lr_scheduler_cfg = lr_scheduler_cfg
        self.action_dim = env_params['action_shape']
        self.normalizer = normalizer

        actor_cfg["action_seq_len"] = action_seq_len
        actor_cfg.update(env_params)
        self.actor = build_model(actor_cfg)
        nn_cfg.update(dict(global_cond_dim=self.obs_feature_dim))
        self.model = build_model(nn_cfg)

        self.horizon = self.action_seq_len = action_seq_len
        self.observation_shape = env_params['obs_shape']

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

        self.actor_optim = build_optimizer([self.model,self.obs_encoder], optim_cfg)
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

        self.init_normalizer = False


    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory
model
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
            epsilon_cond = self.model(x, t, local_cond, global_cond, returns, use_dropout=False)
            epsilon_uncond = self.model(x, t, local_cond, global_cond, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, t, local_cond, global_cond)

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
    def p_sample(self, x, t, local_cond=None, global_cond=None, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, local_cond=local_cond, global_cond=global_cond, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, verbose=True, return_diffusion=False
    ):
        device = self.betas.device

        batch_size = cond_data.shape[0]
        x = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device
        )

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            # 1. apply conditioning
            x[cond_mask] = cond_data[cond_mask]

            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # 2. predict model output and replace sample
            x = self.p_sample(x, timesteps, local_cond, global_cond, returns)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, action_seq_len=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        # horizon = action_seq_len or self.action_seq_len
        # batch_size = len(list(cond_data.values())[0])
        # shape = (batch_size, horizon, self.action_dim) # cond_data.shape
        return self.p_sample_loop(cond_data, cond_mask, local_cond, global_cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, actions, t, cond_mask, local_cond=None, global_cond=None, returns=None):
        noise = torch.randn_like(actions)
        action_noisy = self.q_sample(x_start=actions, t=t, noise=noise)
        # apply conditioning
        action_noisy[cond_mask] = actions[cond_mask]

        pred = self.model(action_noisy, t, local_cond, global_cond, returns)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, actions)

        return loss, {"action_diff_loss": loss.detach().cpu()}

    def loss(self, x, masks, cond_mask, local_cond=None, global_cond=None, returns=None):
        # x is the action, with shape (bs, horizon, act_dim)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(x, t, cond_mask, local_cond, global_cond, returns)
        diffuse_loss = (diffuse_loss * masks.unsqueeze(-1)).mean()
        return diffuse_loss, info

    def forward(self, observation, returns_rate=0.9, mode="eval", *args, **kwargs):
        observation = observation.to_torch(device=self.device, dtype="float32", non_blocking=True)
        
        action_history = observation["actions"]
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")

        if self.obs_as_global_cond:
            act_mask, obs_mask = self.mask_generator((bs, self.horizon, self.action_dim), self.device)
        else:
            raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")
        
        self.set_mode(mode=mode)
        
        # for obs_key in observation.keys():
        #     print(obs_key, observation[obs_key].shape)

        supp = torch.zeros(
            size=[bs, self.action_seq_len-hist_len, self.action_dim], 
            dtype=action_history.dtype,
            device=action_history.device
        )
        action_history = torch.concat([action_history, supp], dim=1)

        obs_fea = self.obs_encoder(observation) # No need to mask out since the history is set as the desired length\

        pred_action_seq = self.conditional_sample(cond_data=action_history, cond_mask=act_mask, global_cond=obs_fea, *args, **kwargs)
        pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
        
        return pred_action_seq[-(self.self.action_seq_len-hist_len)]
    
    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            data = memory.get_all("actions")
            self.normalizer.fit(data, last_n_dims=1, mode='limits')
            self.init_normalizer = True

        batch_size = self.batch_size
        sampled_batch = memory.sample(batch_size).to_torch(dtype="float32")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True) # ["obs","actions"]
        self.actor_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)], 
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1), 
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        # generate impainting mask
        if self.obs_as_global_cond:
            traj_data = sampled_batch["actions"]
            act_mask, obs_mask = self.mask_generator(traj_data.shape, self.device)
        else:
            raise NotImplementedError("Not support diffuse over obs! Please set obs_as_global_cond=True")
        
        masked_obs = sampled_batch['obs']
        for key in masked_obs:
            if isinstance(masked_obs[key], list):
                masked_obs[key] = masked_obs[key][0]
            masked_obs[key] = masked_obs[key][:,obs_mask,...]

        obs_fea = self.obs_encoder(masked_obs)
        traj_data = self.normalizer.normalize(traj_data)

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
    
    ## Not implement yet
    # def step_ema(self):
    #     if self.step < self.step_start_ema:
    #         self.reset_parameters()
    #         return
    #     self.ema.update_model_average(self.ema_model, self.model)

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