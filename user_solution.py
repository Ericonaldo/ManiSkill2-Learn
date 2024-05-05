# REMEMBER: Move this file directly under ManiSkill2-Learn/

import numpy as np
import torch
from gym import spaces
from mani_skill2.evaluation.solution import BasePolicy

from maniskill2_learn.env import build_env, get_env_info
from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)
from maniskill2_learn.utils.data import GDict, is_not_null, to_np
from maniskill2_learn.utils.meta import Config, get_logger
from maniskill2_learn.utils.torch import BaseAgent, load_checkpoint

model_dict = {
    "PickCube-v0": "PickCube-v0/DiffAgent/rgbd/models/model_290000.ckpt",
    # "PickCube-v0": "PickCube-v0/DiffAgent/pcd/models/model_790000.ckpt",
    "PickSingleYCB-v0": "PickCube-v0/DiffAgent/rgbd/models/model_235000.ckpt",
    "StackCube-v0": "StackCube-v0/DiffAgent/rgbd/models/model_245000.ckpt",
    "AssemblingKits-v0": "AssemblingKits-v0/DiffAgent/rgbd/models/model_180000.ckpt",
    "PandaAvoidObstacles-v0": "PandaAvoidObstacles-v0/DiffAgent/rgbd/models/model_160000.ckpt",
    "PegInsertionSide-v0": "PegInsertionSide-v0/DiffAgent/rgbd/models/model_270000.ckpt",
    "PickClutterYCB-v0": "PickClutterYCB-v0/DiffAgent/rgbd/models/model_320000.ckpt",
    "PickSingleEGAD-v0": "PickSingleEGAD-v0/DiffAgent/rgbd/models/model_260000.ckpt",
    "PlugCharger-v0": "PlugCharger-v0/DiffAgent/rgbd/models/model_235000.ckpt",
}


class UserPolicy(BasePolicy):
    def __init__(
        self, env_id: str, observation_space: spaces.Space, action_space: spaces.Space
    ):
        super().__init__(env_id, observation_space, action_space)
        # cfg = Config.fromfile("/root/ManiSkill2-Learn/configs/brl/diff/rgbd_eval.py") # Change this
        # model_path = f'/root/ManiSkill2-Learn/logs/{model_dict[env_id]}' # Change this
        cfg = Config.fromfile("./configs/brl/diff/rgbd_eval.py")  # Change this
        # cfg = Config.fromfile("./configs/brl/diff/pcd_eval.py") # Change this
        model_path = f"./logs/{model_dict[env_id]}"  # Change this
        self.device = "cuda:0"

        self.logger = get_logger()

        cfg.env_cfg["env_name"] = env_id
        cfg.env_cfg["obs_mode"] = self.get_obs_mode(env_id)
        cfg.env_cfg["control_mode"] = self.get_control_mode(env_id)
        cfg.env_cfg["obs_frame"] = "ee"  # Change this
        # if 'Pick' in env_id: # for environments that provide goal locations
        #     cfg.env_cfg["n_goal_points"] = 50 # Change this

        env_params = get_env_info(cfg.env_cfg)
        cfg.agent_cfg["env_params"] = env_params
        obs_shape = env_params["obs_shape"]
        action_shape = env_params["action_shape"]

        if is_not_null(obs_shape) or is_not_null(action_shape):
            replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
            cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)

        print("Final config", cfg)

        self.env = build_env(cfg.env_cfg)

        self.agent = build_agent(cfg.agent_cfg)
        self.agent = self.agent.float().to(self.device)
        load_checkpoint(
            self.agent, model_path, self.device, keys_map=None, logger=self.logger
        )
        self.agent.eval()
        self.agent.set_mode("test")

    def reset(self, observations):
        """Called at the beginning of an episode."""
        self.env.init_queue()

    def act(self, observations):
        observations = self.env.observation(observations)
        observations = self.env.update_observation_queue(observations)
        for key in observations:
            observations[key] = np.expand_dims(observations[key], 0)
        # observations = GDict(observations).unsqueeze(0).to_torch(device=self.device)
        with self.agent.no_sync(mode="actor"):
            action = self.agent(observations, mode="eval")
            action = to_np(action)[0]
        self.env.update_action_queue(action)
        return action

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        # Change this
        # return "pointcloud"
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        # Change this
        return "pd_ee_delta_pose"
