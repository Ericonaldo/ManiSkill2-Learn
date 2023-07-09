"""
Diffusion Policy
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maniskill2_learn.networks import build_model
from maniskill2_learn.networks.modules.block_utils import SimpleMLP as MLP
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.torch import BaseAgent, get_mean_lr, build_optimizer
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.diffusion.mask_generator import LowdimMaskGenerator
from maniskill2_learn.utils.diffusion.normalizer import LinearNormalizer

from ..builder import BRL


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).mean(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


@BRL.register_module()
class ClipAgent(BaseAgent):
    def __init__(
        self,
        visual_nn_cfg,
        optim_cfg,
        nn_cfg,
        actor_cfg,
        env_params,
        action_seq_len,
        eval_action_len=1,
        pcd_cfg=None,
        lr_scheduler_cfg=None,
        batch_size=256,
        agent_share_noise=False,
        obs_as_global_cond=True,  # diffuse action or take obs as condition inputs
        action_visible=True,  # If we cond on some hist actions
        fix_obs_steps=True,  # Randomly cond on certain obs steps or deterministicly
        n_obs_steps=3,
        action_embed_dim=256,
        action_hidden_dims=[256, 512],
        temperature=1.0,
        normalizer=LinearNormalizer(),
        model_type="representation",  # ["representation", "policy"]
        clip_loss_weight=1.0,
        use_simple_clip_target=False,
        use_bc_loss=False,
        bc_loss_type="mse_loss",
        **kwargs,
    ):
        super(ClipAgent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.use_bc_loss = use_bc_loss
        self.bc_loss_type = bc_loss_type
        self.clip_loss_weight = clip_loss_weight
        self.use_simple_clip_target = use_simple_clip_target

        if pcd_cfg is not None:
            visual_nn_cfg["pcd_model"] = build_model(pcd_cfg)
        visual_nn_cfg["n_obs_steps"] = n_obs_steps
        self.obs_encoder = build_model(visual_nn_cfg)
        self.obs_feature_dim = self.obs_encoder.out_feature_dim

        lr_scheduler_cfg = lr_scheduler_cfg
        self.action_dim = env_params["action_shape"]
        self.normalizer = normalizer
        self.model_type = model_type

        self.act_encoder = lambda x: x
        if model_type == "representation":
            assert use_bc_loss is False
            self.act_encoder = MLP(
                input_dim=self.action_dim * action_seq_len,
                output_dim=self.obs_feature_dim,
                hidden_dims=action_hidden_dims,
            )
        else:
            nn_cfg["mlp_spec"][0] = self.obs_feature_dim
            self.action_model = build_model(nn_cfg)

        actor_cfg["action_seq_len"] = action_seq_len
        actor_cfg.update(env_params)
        self.actor = build_model(actor_cfg)

        self.horizon = self.action_seq_len = action_seq_len
        self.observation_shape = env_params["obs_shape"]

        self.agent_share_noise = agent_share_noise

        self.step = 0

        if model_type == "representation":
            self.actor_optim = build_optimizer(
                [self.act_encoder, self.obs_encoder], optim_cfg
            )
        else:
            self.actor_optim = build_optimizer(
                [self.obs_encoder, self.action_model], optim_cfg
            )
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

        self.act_mask, self.obs_mask = None, None

    def eval(self):
        return super().eval()

    def forward(self, observation, returns_rate=0.9, mode="eval", *args, **kwargs):
        assert self.model_type == "policy", "only policy type CLIP agent can be used as policy"
        observation = to_torch(observation, device=self.device, dtype=torch.float32)

        action_history = observation["actions"]
        action_history = self.normalizer.normalize(action_history)
        bs = action_history.shape[0]
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

        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:, obs_mask, ...]

        obs_fea = self.obs_encoder(
            observation
        )  # No need to mask out since the history is set as the desired length

        res = obs_fea
        if self.model_type == "policy":
            action = self.action_model(obs_fea)
            res = self.normalizer.unnormalize(action)

        return res

    def compute_clip_loss(self, obs_fea, act_fea):
        logits = (act_fea @ obs_fea.T) / self.temperature
        if self.use_simple_clip_target:
            targets = torch.arange(logits.shape[0], device=logits.device)
        else:
            obs_similarity = obs_fea @ obs_fea.T
            act_similarity = act_fea @ act_fea.T
            targets = F.softmax(
                (obs_similarity + act_similarity) / 2 * self.temperature, dim=-1
            )
        act_loss = cross_entropy(logits, targets, reduction="none")
        obs_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = (obs_loss + act_loss) / 2.0  # shape: (batch_size)
        loss = loss.mean()
        return loss, {"clip_loss": loss.item()}

    def compute_regression_loss(self, pred_action, target_action, mask=None):
        if mask is None:
            mask = torch.ones_like(target_action[..., 0])
        assert mask.ndim == target_action.ndim - 1
        print_dict = {}
        print_dict["abs_err"] = ((torch.abs(pred_action - target_action).mean(-1) * mask).sum() / mask.sum()).item()
        if hasattr(F, self.bc_loss_type):
            assert self.bc_loss_type in ["mse_loss", "l1_loss", "smooth_l1_loss"]
            actor_loss = (getattr(F, self.bc_loss_type)(pred_action, target_action, reduction="none").mean(-1) * mask).sum() / mask.sum()
            print_dict[f"{self.bc_loss_type}"] = actor_loss.item()
        else:
            raise NotImplementedError(f"BC loss type {self.bc_loss_type} not implemented!")

        return actor_loss, print_dict

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
        masked_obs = sampled_batch["obs"]
        # traj_data = self.normalizer.normalize(traj_data)
        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask = self.mask_generator(traj_data.shape, self.device)
                self.act_mask, self.obs_mask = act_mask, obs_mask
                for key in masked_obs:
                    masked_obs[key] = masked_obs[key][:, obs_mask, ...]
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        obs_fea = self.obs_encoder(masked_obs)
        if self.model_type == "representation":
            act_fea = self.act_encoder(traj_data.reshape(traj_data.shape[0], -1))
        elif self.model_type == "policy":
            obs_fea = self.action_model(obs_fea)
            act_fea = traj_data[:, obs_mask, ...][:, -1]  # (B, action_dim)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented!")

        clip_loss, ret_dict = self.compute_clip_loss(obs_fea, act_fea)
        loss = clip_loss * self.clip_loss_weight
        if self.use_bc_loss:
            bc_loss, bc_ret_dict = self.compute_regression_loss(obs_fea, act_fea)
            ret_dict = {**ret_dict, **bc_ret_dict}
            loss += bc_loss
        loss.backward()
        self.actor_optim.step()

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
        ret_dict = {"clip/" + key: val for key, val in ret_dict.items()}

        self.step += 1

        return ret_dict
