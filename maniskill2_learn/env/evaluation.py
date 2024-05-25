import glob
import logging
import os
import os.path as osp
import shutil
import sys
import json
from collections import deque
from typing import Optional

import cv2
import h5py
import numpy as np
from h5py import File

from maniskill2_learn.methods.brl import RiemannDiffAgent, KPamDiffAgent
from maniskill2_learn.utils.data import (
    DictArray,
    GDict,
    concat_list,
    dict_to_str,
    is_str,
    num_to_str,
    to_item,
    to_np,
)
from maniskill2_learn.utils.file import dump, load, merge_h5_trajectory
from maniskill2_learn.utils.math import split_num
from maniskill2_learn.utils.meta import (
    Worker,
    get_logger,
    get_logger_name,
    get_meta_info,
    get_total_memory,
)

from .builder import EVALUATIONS
from .env_utils import build_env, build_vec_env, get_max_episode_steps, true_done
from .replay_buffer import ReplayMemory


def save_eval_statistics(folder, logger=None, **kwargs):
    if logger is None:
        logger = get_logger()
    lengths = kwargs.get("lengths", None)
    rewards = kwargs.get("rewards", None)
    finishes = kwargs.get("finishes", None)
    if rewards is not None:
        logger.info(
            f"Num of trails: {len(lengths):.2f}, "
            f"Length: {np.mean(lengths):.2f}\u00B1{np.std(lengths):.2f}, "
            f"Reward: {np.mean(rewards):.2f}\u00B1{np.std(rewards):.2f}, "
            f"Success or Early Stop Rate: {np.mean(finishes):.2f}\u00B1{np.std(finishes):.2f}"
        )
        # Some times logger info can be refreshed
        print(
            "\n"
            f"Num of trails: {len(lengths):.2f}, "
            f"Length: {np.mean(lengths):.2f}\u00B1{np.std(lengths):.2f}, "
            f"Reward: {np.mean(rewards):.2f}\u00B1{np.std(rewards):.2f}, "
            f"Success or Early Stop Rate: {np.mean(finishes):.2f}\u00B1{np.std(finishes):.2f}"
        )
        if folder is not None:
            table = [["length", "reward", "finish"]]
            table += [
                [num_to_str(__, precision=2) for __ in _]
                for _ in zip(lengths, rewards, finishes)
            ]
            dump(table, osp.join(folder, "statistics.csv"))
    else:
        action_diff = kwargs.get("action_diff", None)
        if action_diff is None:
            return
        num = kwargs.get("num", -1)
        if action_diff is not None:
            logger.info(
                f"Num of Samples: {num:.2f}, "
                f"Action Difference: {np.mean(action_diff):.2f}\u00B1{np.std(action_diff):.2f}"
            )
        if folder is not None:
            table = [["action_diff"]]
            table += [
                [num_to_str(__, precision=2) for __ in _] for _ in zip(action_diff)
            ]
            dump(table, osp.join(folder, "statistics.csv"))


CV_VIDEO_CODES = {
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
}


def log_mem_info(logger):
    import torch

    from maniskill2_learn.utils.torch import get_cuda_info

    print_dict = {}
    print_dict["memory"] = get_total_memory("G", False)
    print_dict.update(
        get_cuda_info(device=torch.cuda.current_device(), number_only=False)
    )
    print_info = dict_to_str(print_dict)
    logger.info(f"Resource usage: {print_info}")


@EVALUATIONS.register_module()
class FastEvaluation:
    def __init__(
        self, env_cfg=None, num_procs=1, seed=None, eval_action_len=1, **kwargs
    ):
        self.n = num_procs
        self.vec_env = build_vec_env(env_cfg, num_procs, **kwargs, seed=seed)
        self.vec_env.reset()

        self.num_envs = self.vec_env.num_envs
        self.all_env_indices = np.arange(self.num_envs, dtype=np.int32)
        self.log_every_episode = kwargs.get("log_every_episode", True)
        self.log_every_step = kwargs.get("log_every_step", False)

        self.save_traj = kwargs.get("save_traj", False)
        self.save_video = kwargs.get("save_video", False)
        self.only_save_success_traj = kwargs.get("only_save_success_traj", False)

        self.sample_mode = kwargs.get("sample_mode", "eval")

        self.video_format = kwargs.get("video_format", "mp4")
        self.video_fps = kwargs.get("fps", 20)

        self.render_mode = (kwargs.get("render_mode", "cameras"),)  # "rgb_array",

        logger_name = get_logger_name()
        self.logger = get_logger("Evaluation-" + logger_name, with_stream=True)
        self.logger.info(
            f"Evaluation environments have seed in [{seed}, {seed + num_procs})!"
        )

    def reset_pi(self, pi, idx):
        """When we run CEM, we need the level of the rollout env to match the level of test env."""
        if not hasattr(pi, "reset"):
            return
        reset_kwargs = {}
        if hasattr(self.vec_env.vec_env.single_env, "level"):
            reset_kwargs["level"] = self.vec_env.level
        pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

    def run(self, pi, num=1, work_dir=None, **kwargs):
        self.logger.info(f"We will evaluate over {num} episodes!")

        if osp.exists(work_dir):
            self.logger.warning(
                f"We will overwrite this folder {work_dir} during evaluation!"
            )
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        if self.save_video and self.render_mode != "human":
            video_dir = osp.join(work_dir, "videos")
            self.logger.info(f"Save videos to {video_dir}.")
            os.makedirs(video_dir, exist_ok=True)

        if self.save_traj:
            trajectory_path = osp.join(work_dir, "trajectory.h5")
            if osp.exists(trajectory_path):
                self.logger.warning(
                    f"We will overwrite this file {trajectory_path} during evaluation!"
                )
            h5_file = File(trajectory_path, "w")
            self.logger.info(f"Save trajectory at {trajectory_path}.")
            group = h5_file.create_group("meta")
            GDict(get_meta_info()).to_hdf5(group)

        import torch

        num_finished, num_start, num_envs = 0, 0, min(self.num_envs, num)
        traj_idx = np.arange(num_envs, dtype=np.int32)
        video_writers, episodes = None, None

        obs_all = self.vec_env.reset(idx=np.arange(num_envs))
        obs_all = DictArray(obs_all).copy()
        self.reset_pi(pi, self.all_env_indices)

        if self.save_video and self.render_mode != "human":
            video_writers = []
            imgs = self.vec_env.render(mode=self.render_mode, idx=np.arange(num_envs))[
                ..., ::-1
            ]
            for i in range(num_envs):
                video_file = osp.join(video_dir, f"{i}.{self.video_format}")
                video_writers.append(
                    cv2.VideoWriter(
                        video_file,
                        CV_VIDEO_CODES[self.video_format],
                        self.video_fps,
                        (imgs[i].shape[1], imgs[i].shape[0]),
                    )
                )
        else:
            self.vec_env.render(mode=self.render_mode)
        episodes = [[] for i in range(num_envs)]
        num_start = num_envs
        episode_lens, episode_rewards, episode_finishes = (
            np.zeros(num, dtype=np.int32),
            np.zeros(num, dtype=np.float32),
            np.zeros(num, dtype=np.bool_),
        )
        while num_finished < num:
            idx = np.nonzero(traj_idx >= 0)[0]
            obs = obs_all.slice(idx, wrapper=False)
            with torch.no_grad():
                with pi.no_sync(mode="actor"):
                    action = pi(obs, mode=self.sample_mode)
                    action = to_np(action)

            env_state = self.vec_env.get_env_state()
            infos = self.vec_env.step_dict(action, idx=idx, restart=False)
            next_env_state = self.vec_env.get_env_state()
            for key in next_env_state:
                env_state["next_" + key] = next_env_state[key]
            infos.update(env_state)

            infos = GDict(infos).to_array().to_two_dims()
            episode_dones = infos["episode_dones"]
            obs_all.assign(idx, infos["next_obs"])

            if self.log_every_step and self.num_envs == 1:
                reward, done, info, episode_done = GDict(
                    [
                        infos["rewards"],
                        infos["dones"],
                        infos["infos"],
                        infos["episode_dones"],
                    ]
                ).item(wrapper=False)
                assert isinstance(info, dict)
                info_str = dict_to_str(
                    {key.split("/")[-1]: val for key, val in info.items()}
                )
                self.logger.info(
                    f"Episode {traj_idx[0]}, Step {episode_lens[traj_idx[0]]}: Reward: {reward:.3f}, Early Stop or Finish: {done}, Info: {info_str}"
                )
            if self.save_video and self.render_mode != "human":
                imgs = self.vec_env.render(mode=self.render_mode, idx=idx)[..., ::-1]
                for j, i in enumerate(idx):
                    video_writers[i].write(imgs[j])
            else:
                self.vec_env.render(mode=self.render_mode)[0, ..., ::-1]
            reset_idx = []
            reset_levels = []
            for j, i in enumerate(idx):
                episodes[i].append(GDict(infos).slice(j, wrapper=False))
                episode_lens[traj_idx[i]] += 1
                episode_rewards[traj_idx[i]] += to_item(infos["rewards"][j])
                if to_item(episode_dones[j]):
                    num_finished += 1
                    if self.save_video and self.render_mode != "human":
                        video_writers[i].release()

                    episodes_i = GDict.stack(episodes[i], 0)
                    episodes[i] = []

                    reward = episodes_i["rewards"].sum()
                    done = to_item(infos["dones"][j])
                    episode_finishes[traj_idx[i]] = done

                    if self.log_every_episode:
                        self.logger.info(
                            f"Episode {traj_idx[i]} ends: Length {episode_lens[traj_idx[i]]}, Reward: {reward}, Early Stop or Finish: {done}!"
                        )
                        log_mem_info(self.logger)

                    if self.save_traj and (not self.only_save_success_traj or done):
                        group = h5_file.create_group(f"traj_{traj_idx[i]}")
                        GDict(episodes_i.memory).to_hdf5(group)

                    if num_start < num:
                        traj_idx[i] = num_start
                        reset_idx.append(i)
                        num_start += 1
                    else:
                        traj_idx[i] = -1

            reset_idx = np.array(reset_idx, dtype=np.int32)
            if len(reset_idx) > 0:
                obs = self.vec_env.reset(idx=reset_idx)
                obs_all.assign(reset_idx, obs)
                self.reset_pi(pi, reset_idx)

                if self.save_traj:
                    imgs = self.vec_env.render(mode="cameras", idx=reset_idx)[..., ::-1]
                    for j, i in enumerate(reset_idx):
                        video_file = osp.join(
                            video_dir, f"{traj_idx[i]}.{self.video_format}"
                        )
                        video_writers[i] = cv2.VideoWriter(
                            video_file,
                            CV_VIDEO_CODES[self.video_format],
                            self.video_fps,
                            (imgs[j].shape[1], imgs[j].shape[0]),
                        )

        h5_file.close()
        return dict(
            lengths=self.episode_lens,
            rewards=self.episode_rewards,
            finishes=self.episode_finishes,
        )
        # return episode_lens, episode_rewards, episode_finishes

    def close(self):
        self.vec_env.close()


@EVALUATIONS.register_module()
class Evaluation:
    def __init__(
        self,
        env_cfg,
        logger=None,
        worker_id=None,
        save_traj=True,
        only_save_success_traj=False,
        save_video=True,
        use_hidden_state=False,
        sample_mode="eval",
        render_mode="cameras",  # "human", "rgb_array"
        seed=None,
        eval_action_len=1,
        **kwargs,
    ):
        self.vec_env = build_vec_env(env_cfg, seed=seed)
        self.vec_env.reset()
        self.n = 1

        self.horizon = get_max_episode_steps(self.vec_env.single_env)

        self.save_traj = save_traj
        self.only_save_success_traj = only_save_success_traj
        self.save_video = save_video
        self.vec_env_name = env_cfg.env_name
        self.worker_id = worker_id

        self.video_format = kwargs.get("video_format", "mp4")
        self.video_fps = kwargs.get("fps", 20)

        self.log_every_episode = kwargs.get("log_every_episode", True)
        self.log_every_step = kwargs.get("log_every_step", False)

        self.logger = logger
        if logger is None:
            logger_name = get_logger_name()
            log_level = (
                logging.INFO
                if (
                    kwargs.get("log_all", False)
                    or self.worker_id is None
                    or self.worker_id == 0
                )
                else logging.ERROR
            )
            worker_suffix = (
                "-env" if self.worker_id is None else f"-env-{self.worker_id}"
            )

            self.logger = get_logger(
                "Evaluation-" + logger_name + worker_suffix, log_level=log_level
            )
            self.logger.info(f"The Evaluation environment has seed in {seed}!")

        self.use_hidden_state = use_hidden_state
        self.sample_mode = sample_mode

        self.work_dir, self.video_dir, self.trajectory_path = None, None, None
        self.h5_file = None

        self.episode_id = 0
        self.level_index = 0
        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        self.recent_obs = None

        self.data_episode = None
        self.video_writer = None
        self.video_file = None

        self.render_mode = render_mode

        assert not (
            self.use_hidden_state and worker_id is not None
        ), "Use hidden state is only for CEM evaluation!!"
        assert self.horizon is not None and self.horizon, f"{self.horizon}"
        assert (
            self.worker_id is None or not use_hidden_state
        ), "Parallel evaluation does not support hidden states!"
        if save_video and self.render_mode != "human":
            # Use rendering with use additional 1Gi memory in sapien
            image = self.vec_env.render(self.render_mode)[0, ..., ::-1]
            self.logger.info(f"Size of image in the rendered video {image.shape}")
        else:
            self.vec_env.render(self.render_mode)

        self.eval_action_queue = None
        self.eval_action_len = eval_action_len
        if self.eval_action_len > 1:
            self.eval_action_queue = deque(maxlen=self.eval_action_len - 1)

    def start(self, work_dir=None):
        if work_dir is not None:
            self.work_dir = (
                work_dir
                if self.worker_id is None
                else os.path.join(work_dir, f"thread_{self.worker_id}")
            )
            # shutil.rmtree(self.work_dir, ignore_errors=True)
            os.makedirs(self.work_dir, exist_ok=True)
            if self.save_video and self.render_mode != "human":
                self.video_dir = osp.join(self.work_dir, "videos")
                os.makedirs(self.video_dir, exist_ok=True)
            if self.save_traj:
                self.trajectory_path = osp.join(self.work_dir, "trajectory.h5")
                self.h5_file = File(self.trajectory_path, "w")
                self.logger.info(f"Save trajectory at {self.trajectory_path}.")
                group = self.h5_file.create_group("meta")
                GDict(get_meta_info()).to_hdf5(group)

        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.recent_obs = None
        self.data_episode = None
        self.video_writer = None
        self.level_index = -1
        self.logger.info(f"Begin to evaluate in worker {self.worker_id}")
        if self.worker_id is not None:
            print(f"Begin to evaluate in worker {self.worker_id}", flush=True)

        self.episode_id = -1
        if hasattr(self, "seed_list"):
            seed = self.seed_list[self.seed_idx]
            self.seed_idx = self.seed_idx + 1
        else:
            seed = None
        self.reset(seed)

    def done(self):
        self.episode_lens.append(self.episode_len)
        self.episode_rewards.append(self.episode_reward)
        self.episode_finishes.append(self.episode_finish)

        if self.save_traj and self.data_episode is not None:
            if (not self.only_save_success_traj) or (
                self.only_save_success_traj and self.episode_finish
            ):
                group = self.h5_file.create_group(f"traj_{self.episode_id}")
                self.data_episode.to_hdf5(group, with_traj_index=False)

            self.data_episode = None
        # exit(0)
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.vec_env.seed(seed)
            self.extra_vec_env.seed(seed)

        self.episode_id += 1
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        level = None
        self.recent_obs = self.vec_env.reset()
        if getattr(self, "extra_vec_env", None) is not None:
            self.extra_vec_env.reset()
        if hasattr(self.vec_env, "level"):
            level = self.vec_env.level
        elif hasattr(self.vec_env.unwrapped, "_main_seed"):
            level = self.vec_env.unwrapped._main_seed
        if level is not None and self.log_every_episode:
            extra_output = (
                "" if self.level_index is None else f"with level id {self.level_index}"
            )
            self.logger.info(
                f"Episode {self.episode_id} begins, run on level {level} {extra_output}!"
            )

        self.init_eval()

    def init_eval(self):
        if self.eval_action_queue is not None:
            if self.eval_action_len > 1:
                self.eval_action_queue = deque(maxlen=self.eval_action_len - 1)

    def step(self, action):
        data_to_store = {"obs": self.recent_obs}

        if self.save_traj:
            env_state = self.vec_env.get_env_state()
            data_to_store.update(env_state)

        if self.save_video and self.render_mode != "human":
            image = self.vec_env.render(mode=self.render_mode)[0, ..., ::-1]
            image = image.astype(np.uint8).copy()
            image = cv2.putText(
                image,
                "Current action: " + str(action[0]),
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            # # HACK(zbzhu)
            # image = cv2.putText(
            #     image,
            #     "Current relative pose: " + str(np.round(self.recent_obs["state"][0, -1, -7:], 2)),
            #     (20, 320),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 0),
            #     1,
            #     cv2.LINE_AA,
            # )

            if self.video_writer is None:
                self.video_file = osp.join(
                    self.video_dir, f"{self.episode_id}.{self.video_format}"
                )

                self.video_writer = cv2.VideoWriter(
                    self.video_file,
                    CV_VIDEO_CODES[self.video_format],
                    self.video_fps,
                    (image.shape[1], image.shape[0]),
                )
            self.video_writer.write(image)
        else:
            self.vec_env.render(mode=self.render_mode)
        infos = self.vec_env.step_dict(action, restart=False)

        reward, done, info, episode_done = GDict(
            [infos["rewards"], infos["dones"], infos["infos"], infos["episode_dones"]]
        ).item(wrapper=False)
        self.episode_len += 1
        self.episode_reward += float(reward)
        if self.log_every_step:
            assert isinstance(info, dict)
            info_str = dict_to_str(
                {key.split("/")[-1]: val for key, val in info.items()}
            )
            self.logger.info(
                f"Episode {self.episode_id}, Step {self.episode_len}: Reward: {reward:.3f}, Early Stop or Finish: {done}, Info: {info_str}"
            )
            if self.worker_id is not None:
                print(
                    f"Woker ID {self.worker_id}, Episode {self.episode_id}, Step {self.episode_len}: Reward: {reward:.3f}, Early Stop or Finish: {done}, Info: {info_str}",
                    flush=True,
                )

        if self.save_traj:
            data_to_store.update(infos)
            next_env_state = self.vec_env.get_env_state()
            for key in next_env_state:
                data_to_store[f"next_{key}"] = next_env_state[key]
            # if self.data_episode is None: # This trick is problematic for ManiSkill 2022
            if self.data_episode is None:
                self.data_episode = ReplayMemory(self.horizon)
            data_to_store = GDict(data_to_store).to_array().f64_to_f32().to_two_dims()
            self.data_episode.push_batch(data_to_store)

        if episode_done:
            if self.save_video and self.render_mode != "human":
                image = self.vec_env.render(mode=self.render_mode)[0, ..., ::-1]
                image = image.astype(np.uint8).copy()
                image = cv2.putText(
                    image,
                    "Current action: " + str(action[0]),
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                self.video_writer.write(image)
            else:
                self.vec_env.render(mode=self.render_mode)
            if self.log_every_episode:
                self.logger.info(
                    f"Episode {self.episode_id} ends: Length {self.episode_len}, Reward: {self.episode_reward}, Early Stop or Finish: {done}"
                )
                if self.worker_id is not None:
                    print(
                        f"Woker ID {self.worker_id}, Episode {self.episode_id} ends: Length {self.episode_len}, Reward: {self.episode_reward}, Early Stop or Finish: {done}",
                        flush=True,
                    )
            self.episode_finish = done
            self.done()
            if hasattr(self, "seed_list"):
                seed = self.seed_list[self.seed_idx]
                self.seed_idx = self.seed_idx + 1
            else:
                seed = None
            self.reset(seed)
        else:
            self.recent_obs = infos["next_obs"]
        return self.recent_obs, episode_done

    def finish(self):
        if self.save_traj:
            self.h5_file.close()

    def run(self, pi, num=1, work_dir=None, **kwargs):
        self.start(work_dir)
        import torch

        replay = None
        if "memory" in kwargs:
            replay = kwargs["memory"]

        def reset_pi():
            if hasattr(pi, "reset"):
                assert (
                    self.worker_id is None
                ), "Reset policy only works for single thread!"
                reset_kwargs = {}
                if hasattr(self.vec_env, "level"):
                    # When we run CEM, we need the level of the rollout env to match the level of test env.
                    reset_kwargs["level"] = self.vec_env.level
                pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

        reset_pi()
        recent_obs = self.recent_obs

        while self.episode_id < num:
            if isinstance(pi, KPamDiffAgent) and pi.stage != "kpam":
                kpam_obs = self.vec_env.get_obs_kpam()
                if pi.is_preinserted(kpam_obs):
                    pi.kpam()

            if isinstance(pi, KPamDiffAgent) and pi.stage == "kpam":
                kpam_obs = self.vec_env.get_obs_kpam()
                action = pi(kpam_obs).reshape(1, -1)
            else:
                if self.eval_action_queue is not None and len(self.eval_action_queue):
                    action = self.eval_action_queue.popleft()
                else:
                    if self.use_hidden_state:
                        recent_obs = self.vec_env.get_state()
                    with torch.no_grad():
                        with pi.no_sync(
                            mode=["actor", "model", "obs_encoder"], ignore=True
                        ):
                            if (
                                isinstance(pi, KPamDiffAgent)
                                and self.vec_env.is_grasped().item()
                            ):
                                kpam_obs = self.vec_env.get_obs_kpam()
                                action = pi(
                                    recent_obs,
                                    kpam_obs=kpam_obs,
                                    mode=self.sample_mode,
                                    memory=replay,
                                )
                            else:
                                action = pi(
                                    recent_obs, mode=self.sample_mode, memory=replay
                                )
                            action = to_np(action)
                            if (
                                (self.eval_action_queue is not None)
                                and (len(self.eval_action_queue) == 0)
                                and self.eval_action_len > 1
                            ):
                                # for i in range(self.eval_action_len-1):
                                for i in range(
                                    min(self.eval_action_len - 1, action.shape[1] - 1)
                                ):  # Allow eval action len to be different with predicted action len
                                    self.eval_action_queue.append(action[:, i + 1, :])
                                action = action[:, 0]

            recent_obs, episode_done = self.step(action)

            if episode_done:
                reset_pi()
                log_mem_info(self.logger)
        self.finish()

        return dict(
            lengths=self.episode_lens,
            rewards=self.episode_rewards,
            finishes=self.episode_finishes,
        )

    def close(self):
        if hasattr(self, "env"):
            del self.vec_env
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()


@EVALUATIONS.register_module()
class BatchEvaluation:
    def __init__(
        self,
        env_cfg,
        num_procs=1,
        save_traj=True,
        save_video=True,
        enable_merge=True,
        sample_mode="eval",
        render_mode="cameras",
        seed=None,
        eval_action_len=1,
        **kwargs,
    ):
        self.work_dir = None
        self.vec_env_name = env_cfg.env_name
        self.save_traj = save_traj
        self.save_video = save_video
        self.num_procs = num_procs
        self.enable_merge = enable_merge
        self.sample_mode = sample_mode
        self.render_mode = render_mode

        self.video_dir = None
        self.trajectory_path = None
        self.recent_obs = None

        self.n = num_procs
        self.workers = []
        log_level = logging.INFO
        self.logger = get_logger("Evaluation-" + get_logger_name(), log_level=log_level)

        seed = seed if seed is not None else np.random.randint(int(1e9))
        self.logger.info(
            f"Evaluation environments have seed in [{seed}, {seed + self.n})!"
        )
        for i in range(self.n):
            self.workers.append(
                Worker(
                    Evaluation,
                    i,
                    logger=self.logger,
                    worker_seed=seed + i,
                    env_cfg=env_cfg,
                    save_traj=save_traj,
                    seed=seed + i,
                    save_video=save_video,
                    render_mode=render_mode,
                    sample_mode=sample_mode,
                    **kwargs,
                )
            )

        self.eval_action_queue = None
        self.eval_action_len = eval_action_len
        if self.eval_action_len > 1:
            self.eval_action_queue = {}
            for i in range(self.n):
                self.eval_action_queue[i] = deque(maxlen=self.eval_action_len - 1)

    def init_eval(self, i):
        if self.eval_action_queue is not None:
            self.eval_action_queue[i] = deque(maxlen=self.eval_action_len - 1)

    def start(self, work_dir=None):
        self.work_dir = work_dir
        if self.enable_merge and self.work_dir is not None:
            # shutil.rmtree(self.work_dir, ignore_errors=True)
            self.video_dir = osp.join(self.work_dir, "videos")
            self.trajectory_path = osp.join(self.work_dir, "trajectory.h5")
        for worker in self.workers:
            worker.call("start", work_dir=work_dir)
        for worker in self.workers:
            worker.wait()

        for i in range(self.n):
            self.workers[i].get_attr("recent_obs")
        self.recent_obs = DictArray.concat(
            [self.workers[i].wait() for i in range(self.n)], axis=0
        )

    @property
    def episode_lens(self):
        for i in range(self.n):
            self.workers[i].get_attr("episode_lens")
        return concat_list([self.workers[i].wait() for i in range(self.n)])

    @property
    def episode_rewards(self):
        for i in range(self.n):
            self.workers[i].get_attr("episode_rewards")
        return concat_list([self.workers[i].wait() for i in range(self.n)])

    @property
    def episode_finishes(self):
        for i in range(self.n):
            self.workers[i].get_attr("episode_finishes")
        return concat_list([self.workers[i].wait() for i in range(self.n)])

    def finish(self):
        for i in range(self.n):
            self.workers[i].call("finish")
        for i in range(self.n):
            self.workers[i].wait()

    def merge_results(self, num_threads):
        if self.save_traj:
            h5_files = [
                osp.join(self.work_dir, f"thread_{i}", "trajectory.h5")
                for i in range(num_threads)
            ]
            merge_h5_trajectory(h5_files, self.trajectory_path)
            self.logger.info(
                f"Merge {len(h5_files)} trajectories to {self.trajectory_path}"
            )
        if self.save_video and self.render_mode != "human":
            index = 0
            os.makedirs(self.video_dir)
            for i in range(num_threads):
                num_traj = len(
                    glob.glob(osp.join(self.work_dir, f"thread_{i}", "videos", "*.mp4"))
                )
                for j in range(num_traj):
                    shutil.copyfile(
                        osp.join(self.work_dir, f"thread_{i}", "videos", f"{j}.mp4"),
                        osp.join(self.video_dir, f"{index}.mp4"),
                    )
                    index += 1
            self.logger.info(f"Merge {index} videos to {self.video_dir}")
        for dir_name in glob.glob(osp.join(self.work_dir, "*")):
            if osp.isdir(dir_name) and osp.basename(dir_name).startswith("thread"):
                shutil.rmtree(dir_name, ignore_errors=True)

    def run(self, pi, num=1, work_dir=None, **kwargs):
        n, running_steps = split_num(num, self.n)
        self.start(work_dir)
        num_finished = [0 for i in range(n)]
        if hasattr(pi, "reset"):
            pi.reset()
        import torch

        while True:
            sys.stdout.flush()
            finish = True
            for i in range(n):
                finish = finish and (num_finished[i] >= running_steps[i])
            if finish:
                break
            if self.eval_action_queue is not None and self.eval_action_len > 1:
                action_dim = self.recent_obs["actions"].shape[-1]
                tmp = np.array([[[None] * action_dim] * self.eval_action_len] * self.n)
                actions = np.array([None] * self.n)

            with torch.no_grad():
                if self.eval_action_queue is not None:
                    for i in range(n):
                        if len(self.eval_action_queue[i]):
                            actions[i] = self.eval_action_queue[i].popleft()
                            if num_finished[i] < running_steps[i]:
                                self.workers[i].call("step", to_np(actions[i : i + 1]))

                none_idx = [i for i, x in enumerate(actions) if x is None]
                if len(none_idx):
                    with pi.no_sync(
                        mode=["actor", "model", "obs_encoder"], ignore=True
                    ):
                        tmp[none_idx] = to_np(
                            pi(
                                dict(self.recent_obs.get(none_idx)),
                                mode=self.sample_mode,
                            )[:, : self.eval_action_len, :]
                        )

                if self.eval_action_queue is not None:
                    for j in range(self.n):
                        if (actions[j] is None) and self.eval_action_len > 1:
                            assert (
                                len(self.eval_action_queue[j]) == 0
                            ), "Why queue not empty?"
                            # for i in range(self.eval_action_len-1):
                            for i in range(
                                min(self.eval_action_len - 1, tmp[j].shape[0] - 1)
                            ):  # Allow eval action len to be different with predicted action len
                                self.eval_action_queue[j].append(tmp[j, i + 1, :])
                            actions[j] = tmp[j, 0, :]
                            if num_finished[j] < running_steps[j]:
                                self.workers[j].call("step", to_np(actions[j : j + 1]))

            # actions = to_np(actions)
            # for i in range(n):
            #     if num_finished[i] < running_steps[i]:
            #         self.workers[i].call("step", actions[i:i+1])
            for i in range(n):
                if num_finished[i] < running_steps[i]:
                    obs_i, episode_done = GDict(self.workers[i].wait()).slice(
                        0, wrapper=False
                    )
                    self.recent_obs.assign((i,), obs_i)
                    num_finished[i] += int(episode_done)
                    if episode_done:
                        self.init_eval(i)
                    # Commenting this out for now; this causes pynvml.nvml.NVMLError_FunctionNotFound for some reason
                    # if i == 0 and bool(episode_done):
                    #     log_mem_info(self.logger)

        self.finish()
        if self.enable_merge:
            self.merge_results(n)

        return dict(
            lengths=self.episode_lens,
            rewards=self.episode_rewards,
            finishes=self.episode_finishes,
        )

    def close(self):
        for worker in self.workers:
            worker.call("close")
            worker.close()


@EVALUATIONS.register_module()
class OfflineDiffusionEvaluation:
    def __init__(
        self,
        env_cfg,
        worker_id=None,
        sample_mode="train",
        seed=None,
        **kwargs,
    ):
        self.n = 1
        self.worker_id = worker_id

        self.log_every_episode = kwargs.get("log_every_episode", True)
        self.log_every_step = kwargs.get("log_every_step", False)

        logger_name = get_logger_name()
        log_level = (
            logging.INFO
            if (
                kwargs.get("log_all", False)
                or self.worker_id is None
                or self.worker_id == 0
            )
            else logging.ERROR
        )
        worker_suffix = "-env" if self.worker_id is None else f"-env-{self.worker_id}"

        self.logger = get_logger(
            "Evaluation-" + logger_name + worker_suffix, log_level=log_level
        )
        self.logger.info(f"The Evaluation environment has seed in {seed}!")

        self.sample_mode = sample_mode

        self.work_dir, self.video_dir, self.trajectory_path = None, None, None
        self.h5_file = None

        self.sample_id = 0
        self.level_index = 0
        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        self.recent_obs = None
        self.obs_mask, self.act_mask = None, None

        self.data_episode = None
        self.video_writer = None
        self.video_file = None

    def start(self, work_dir=None):
        if work_dir is not None:
            self.work_dir = (
                work_dir
                if self.worker_id is None
                else os.path.join(work_dir, f"thread_{self.worker_id}")
            )
            # shutil.rmtree(self.work_dir, ignore_errors=True)
            os.makedirs(self.work_dir, exist_ok=True)

        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.data_episode = None
        self.level_index = -1
        self.logger.info(f"Begin to evaluate in worker {self.worker_id}")

        self.sample_id = -1

    def finish(self):
        return

    def run(self, pi, memory, num=1, work_dir=None, **kwargs):
        self.start(work_dir)
        import torch

        sampled_batch = memory.sample(num, mode="eval")

        if self.act_mask is None or self.obs_mask is None:
            act_mask, obs_mask, _ = pi.mask_generator(
                (num, pi.horizon, pi.action_dim), pi.device
            )
            self.act_mask, self.obs_mask = to_np(act_mask), to_np(obs_mask)

        observation = sampled_batch["obs"]
        if "state" in observation:
            observation["state"] = observation["state"][:, self.obs_mask]
        if "actions" in sampled_batch:
            observation["actions"] = sampled_batch["actions"][:, self.act_mask[0, :, 0]]
        if "timesteps" in sampled_batch:
            observation["timesteps"] = sampled_batch["timesteps"]

        with torch.no_grad():
            with pi.no_sync(mode=["actor", "model", "obs_encoder"], ignore=True):
                action_sequence = pi(observation, mode=self.sample_mode)
                assert (
                    action_sequence.shape == sampled_batch["actions"].shape
                ), "action_sequence shape is {}, yet sampled_batch actions shape is {}".format(
                    action_sequence.shape, sampled_batch["actions"].shape
                )

        action_sequence = action_sequence.cpu().numpy()
        self.action_diff = (
            ((action_sequence - sampled_batch["actions"]) ** 2)
            .mean(axis=-1)
            .mean(axis=-1)
        )

        self.finish()
        return dict(num=num, action_diff=self.action_diff)

    def close(self):
        return


@EVALUATIONS.register_module()
class KPamEvaluation(Evaluation):
    def __init__(self, extra_env_cfg=None, seed=None, *args, **kwargs):
        if extra_env_cfg is not None:
            self.extra_vec_env = build_vec_env(extra_env_cfg, seed=seed)
            self.extra_vec_env.reset()
        else:
            self.extra_vec_env = None
        super().__init__(seed=seed, *args, **kwargs)

    def save_pointcloud_pkl(self, pointcloud_obs):
        from skimage.io import imsave

        env_img = self.vec_env.render(mode=self.render_mode, idx=[0])[0, ..., ::-1]
        imsave("camera_image.png", env_img)

        from maniskill2_learn.methods.kpam.kpam_utils import recursive_squeeze

        kpam_obs = self.vec_env.get_obs_kpam()
        kpam_obs = recursive_squeeze(kpam_obs, axis=0)

        import pickle

        with open("pointcloud.pkl", "wb") as f:
            pickle.dump(
                dict(
                    xyz=pointcloud_obs["xyz"][0],
                    rgb=pointcloud_obs["rgb"][0],
                    seg=pointcloud_obs["gt_seg"][0],
                    kpam_obs=kpam_obs,
                ),
                f,
            )

    def run(self, pi, num=1, work_dir=None, **kwargs):
        self.start(work_dir)
        import torch

        replay = None
        if "memory" in kwargs:
            replay = kwargs["memory"]

        def reset_pi():
            if hasattr(pi, "reset"):
                assert (
                    self.worker_id is None
                ), "Reset policy only works for single thread!"
                reset_kwargs = {}
                if hasattr(self.vec_env, "level"):
                    # When we run CEM, we need the level of the rollout env to match the level of test env.
                    reset_kwargs["level"] = self.vec_env.level
                pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

        reset_pi()
        grasped = self.vec_env.is_grasped().item()
        recent_obs = self.recent_obs

        # if self.extra_vec_env is not None:
        #     env_states = self.vec_env.get_state()
        #     self.extra_vec_env.set_state(env_states)
        #     pointcloud_obs = self.extra_vec_env.get_obs()
        #     self.save_pointcloud_pkl(pointcloud_obs)

        while self.episode_id < num:
            if grasped:
                kpam_obs = self.vec_env.get_obs_kpam()
                if self.extra_vec_env is not None:
                    env_states = self.vec_env.get_state()
                    self.extra_vec_env.set_state(env_states)
                    pointcloud_obs = self.extra_vec_env.get_obs()
                    # self.save_pointcloud_pkl(pointcloud_obs)
                    action = pi(kpam_obs, pointcloud_obs, use_kpam=True).reshape(1, -1)
                else:
                    action = pi(kpam_obs, use_kpam=True).reshape(1, -1)

            else:
                if self.eval_action_queue is not None and len(self.eval_action_queue):
                    action = self.eval_action_queue.popleft()
                else:
                    if self.use_hidden_state:
                        recent_obs = self.vec_env.get_state()
                    with torch.no_grad():
                        with pi.no_sync(
                            mode=["actor", "model", "obs_encoder"], ignore=True
                        ):
                            action = pi(
                                recent_obs,
                                mode=self.sample_mode,
                                memory=replay,
                                use_kpam=False,
                            )
                            action = to_np(action)
                            if (
                                (self.eval_action_queue is not None)
                                and (len(self.eval_action_queue) == 0)
                                and self.eval_action_len > 1
                            ):
                                # for i in range(self.eval_action_len-1):
                                for i in range(
                                    min(self.eval_action_len - 1, action.shape[1] - 1)
                                ):  # Allow eval action len to be different with predicted action len
                                    self.eval_action_queue.append(action[:, i + 1, :])
                                action = action[:, 0]

            recent_obs, episode_done = self.step(action)
            grasped = self.vec_env.is_grasped().item()
            if grasped:
                self.eval_action_queue.clear()

            if episode_done:
                reset_pi()
                log_mem_info(self.logger)
        self.finish()

        return dict(
            lengths=self.episode_lens,
            rewards=self.episode_rewards,
            finishes=self.episode_finishes,
        )


@EVALUATIONS.register_module()
class RiemannEvaluation(Evaluation):
    def __init__(self, extra_env_cfg=None, seed=None, *args, **kwargs):
        if extra_env_cfg is not None:
            self.extra_vec_env = build_vec_env(extra_env_cfg, seed=seed)
            self.extra_vec_env.reset()
        else:
            self.extra_vec_env = None
        super().__init__(seed=seed, *args, **kwargs)

    def save_pointcloud_pkl(self, pointcloud_obs):
        from skimage.io import imsave

        env_img = self.vec_env.render(mode=self.render_mode, idx=[0])[0]
        imsave("camera_image.png", env_img)

        from maniskill2_learn.methods.kpam.kpam_utils import recursive_squeeze

        kpam_obs = self.vec_env.get_obs_kpam()
        kpam_obs = recursive_squeeze(kpam_obs, axis=0)

        import pickle

        with open("pointcloud.pkl", "wb") as f:
            pickle.dump(
                dict(
                    xyz=pointcloud_obs["xyz"][0],
                    rgb=pointcloud_obs["rgb"][0],
                    seg=pointcloud_obs["gt_seg"][0],
                    kpam_obs=kpam_obs,
                ),
                f,
            )

    def run(self, pi, num=1, work_dir=None, **kwargs):
        # HACK(zbzhu): to make the env and extra env the same seed before EVERY reset
        self.seed_idx = 0
        self.seed_list = np.arange(2000, 2000 + num + 1)

        self.start(work_dir)
        import torch

        replay = None
        if "memory" in kwargs:
            replay = kwargs["memory"]

        def reset_pi():
            if hasattr(pi, "reset"):
                assert (
                    self.worker_id is None
                ), "Reset policy only works for single thread!"
                reset_kwargs = {}
                if hasattr(self.vec_env, "level"):
                    # When we run CEM, we need the level of the rollout env to match the level of test env.
                    reset_kwargs["level"] = self.vec_env.level
                pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

        reset_pi()
        grasped = self.vec_env.is_grasped().item()
        recent_obs = self.recent_obs

        # if self.extra_vec_env is not None:
        #     env_states = self.vec_env.get_state()
        #     self.extra_vec_env.set_state(env_states)
        #     pointcloud_obs = self.extra_vec_env.get_obs()
        #     self.save_pointcloud_pkl(pointcloud_obs)

        while self.episode_id < num:
            stage = getattr(pi, "stage", None)
            if isinstance(pi, RiemannDiffAgent) and pi.check_pose_empty():
                env_states = self.vec_env.get_state()
                self.extra_vec_env.set_state(env_states)
                pointcloud_obs = self.extra_vec_env.get_obs()
                dict_obs = self.vec_env.get_obs_kpam()
                pi.predict_riemann_pose(dict_obs, pointcloud_obs)

            if stage == "diffusion":
                dict_obs = self.vec_env.get_obs_kpam()
                # if preinserted, then switch to rule-based control
                if pi.is_preinserted(dict_obs):
                    pi.kpam()
                    stage = getattr(pi, "stage", None)

            if (grasped and stage is None) or stage == "kpam":
                dict_obs = self.vec_env.get_obs_kpam()
                if self.extra_vec_env is not None:
                    env_states = self.vec_env.get_state()
                    self.extra_vec_env.set_state(env_states)
                    pointcloud_obs = self.extra_vec_env.get_obs()
                    # self.save_pointcloud_pkl(pointcloud_obs)
                    action = pi(recent_obs, dict_obs, pointcloud_obs).reshape(
                        1, -1
                    )
                else:
                    action = pi(recent_obs, dict_obs).reshape(1, -1)

            else:
                if self.eval_action_queue is not None and len(self.eval_action_queue):
                    action = self.eval_action_queue.popleft()
                else:
                    if self.use_hidden_state:
                        recent_obs = self.vec_env.get_state()
                    with torch.no_grad():
                        with pi.no_sync(
                            mode=["actor", "model", "obs_encoder"], ignore=True
                        ):
                            if stage == "diffusion" and grasped:
                                dict_obs = self.vec_env.get_obs_kpam()
                                env_states = self.vec_env.get_state()
                                self.extra_vec_env.set_state(env_states)
                                pointcloud_obs = self.extra_vec_env.get_obs()
                                action = pi(
                                    recent_obs,
                                    dict_obs,
                                    pointcloud_obs,
                                    mode=self.sample_mode,
                                    memory=replay,
                                )
                            else:
                                action = pi(
                                    recent_obs,
                                    mode=self.sample_mode,
                                    memory=replay,
                                )

                            action = to_np(action)
                            if (
                                (self.eval_action_queue is not None)
                                and (len(self.eval_action_queue) == 0)
                                and self.eval_action_len > 1
                            ):
                                for i in range(
                                    min(self.eval_action_len - 1, action.shape[1] - 1)
                                ):  # Allow eval action len to be different with predicted action len
                                    self.eval_action_queue.append(action[:, i + 1, :])
                                action = action[:, 0]

            recent_obs, episode_done = self.step(action)
            grasped = self.vec_env.is_grasped().item()
            if grasped and (not isinstance(pi, RiemannDiffAgent) or pi.stage == "kpam"):
                self.eval_action_queue.clear()

            if episode_done:
                reset_pi()
                log_mem_info(self.logger)
        self.finish()

        return dict(
            lengths=self.episode_lens,
            rewards=self.episode_rewards,
            finishes=self.episode_finishes,
        )


@EVALUATIONS.register_module()
class KeyframeEvaluation(Evaluation):
    def __init__(
        self,
        traj_filename: str,
        traj_json_filename: str,
        *args,
        **kwargs,
    ):
        with open(traj_json_filename, "r") as f:
            json_file = json.load(f)
        self.reset_kwargs = {}
        for d in json_file["episodes"]:
            episode_id = d["episode_id"]
            r_kwargs = d["reset_kwargs"]
            # XXX(zbzhu): change 'seed' to np.array to skip the bug of slicing native list in Gdict
            self.reset_kwargs[episode_id] = {"seed": np.array([r_kwargs["seed"]])}
        self.load_keyframes(traj_filename)
        self.ep_keyframes = None
        super().__init__(*args, **kwargs)

    def load_keyframes(self, traj_filename: str):
        traj_file = h5py.File(traj_filename, "r")
        traj_keys = list(traj_file.keys())
        traj_file.close()
        traj_keys = [(int(key.split("_")[1]), int(key.split("_")[2])) for key in traj_keys]
        sorted_traj_keys = sorted(traj_keys, key=lambda x: (x[0], x[1]))

        self.init_env_states = {}
        self.keyframe_obs = {}
        for traj_id, keyframe_id in sorted_traj_keys:
            item = GDict.from_hdf5(traj_filename, keys=f"traj_{traj_id}_{keyframe_id}")
            keyframe = item["actions"][-1]
            if keyframe_id == 0:
                self.init_env_states[traj_id] = item["env_states"][0]
                self.keyframe_obs[traj_id] = [keyframe]
            else:
                self.keyframe_obs[traj_id].append(keyframe)
        self.traj_ids = list(self.init_env_states.keys())

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.vec_env.seed(seed)
            self.extra_vec_env.seed(seed)

        self.episode_id += 1
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False

        reset_kwargs = self.reset_kwargs[self.traj_ids[self.episode_id]]
        init_env_state = self.init_env_states[self.traj_ids[self.episode_id]]
        self.ep_keyframes = self.keyframe_obs[self.traj_ids[self.episode_id]]

        self.vec_env.reset(**reset_kwargs)
        self.vec_env.set_state(init_env_state.reshape(1, -1))
        self.recent_obs = self.vec_env.get_obs()
        if getattr(self, "extra_vec_env", None) is not None:
            self.extra_vec_env.reset(**reset_kwargs)
            self.extra_vec_env.set_state(init_env_state.reshape(1, 1))

        self.init_eval()

    def run(self, pi, num: int = 1, work_dir: str = None, **kwargs):
        assert num <= len(self.init_env_states), f"Episode num {num} should be less than {len(self.init_env_states)}"

        self.start(work_dir)
        import torch

        replay = None
        if "memory" in kwargs:
            replay = kwargs["memory"]

        def reset_pi():
            if hasattr(pi, "reset"):
                assert (
                    self.worker_id is None
                ), "Reset policy only works for single thread!"
                reset_kwargs = {}
                if hasattr(self.vec_env, "level"):
                    # When we run CEM, we need the level of the rollout env to match the level of test env.
                    reset_kwargs["level"] = self.vec_env.level
                pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

        reset_pi()
        recent_obs = self.recent_obs
        keyframe_id = 0

        while self.episode_id < num:
            if self.eval_action_queue is not None and len(self.eval_action_queue):
                action = self.eval_action_queue.popleft()
            else:
                if self.use_hidden_state:
                    recent_obs = self.vec_env.get_state()
                with torch.no_grad():
                    with pi.no_sync(
                        mode=["actor", "model", "obs_encoder"], ignore=True
                    ):
                        keyframe = np.expand_dims(self.ep_keyframes[keyframe_id], axis=0)
                        keyframe_id = min(keyframe_id + 1, len(self.ep_keyframes) - 1)
                        action = pi(
                            recent_obs, next_keyframe=keyframe, mode=self.sample_mode, memory=replay
                        )
                        action = to_np(action)
                        if (
                            (self.eval_action_queue is not None)
                            and (len(self.eval_action_queue) == 0)
                            and self.eval_action_len > 1
                        ):
                            # for i in range(self.eval_action_len-1):
                            for i in range(
                                min(self.eval_action_len - 1, action.shape[1] - 1)
                            ):  # Allow eval action len to be different with predicted action len
                                self.eval_action_queue.append(action[:, i + 1, :])
                            action = action[:, 0]

            recent_obs, episode_done = self.step(action)

            if episode_done:
                reset_pi()
                keyframe_id = 0
                log_mem_info(self.logger)

        self.finish()

        return dict(
            lengths=self.episode_lens,
            rewards=self.episode_rewards,
            finishes=self.episode_finishes,
        )


@EVALUATIONS.register_module()
class OfflineKeyframeEvaluation(OfflineDiffusionEvaluation):
    def run(self, pi, memory, num: int = 1, work_dir: str = None, **kwargs):
        self.start(work_dir)
        import torch

        sampled_batch = memory.sample(num, mode="eval")

        if self.act_mask is None or self.obs_mask is None:
            act_mask, obs_mask, _ = pi.mask_generator(
                (num, pi.horizon, pi.action_dim), pi.device
            )
            self.act_mask, self.obs_mask = to_np(act_mask), to_np(obs_mask)

        observation = sampled_batch["obs"]
        keyframe = sampled_batch["actions"][:, -1]
        if "state" in observation:
            observation["state"] = observation["state"][:, self.obs_mask]
        if "timesteps" in sampled_batch:
            observation["timesteps"] = sampled_batch["timesteps"]

        with torch.no_grad():
            with pi.no_sync(mode=["actor", "model", "obs_encoder"], ignore=True):
                action_sequence = pi(observation, next_keyframe=keyframe, mode=self.sample_mode)
                assert (
                    action_sequence.shape == sampled_batch["actions"].shape
                ), "action_sequence shape is {}, yet sampled_batch actions shape is {}".format(
                    action_sequence.shape, sampled_batch["actions"].shape
                )

        action_sequence = action_sequence.cpu().numpy()
        self.action_diff = (
            ((action_sequence[:, 1:-1] - sampled_batch["actions"][:, 1:-1]) ** 2)
            .mean(axis=-1)
            .mean(axis=-1)
        )

        self.finish()
        return dict(num=num, action_diff=self.action_diff)
