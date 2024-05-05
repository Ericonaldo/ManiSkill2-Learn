"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.

Need to export MS2_ASSET_DIR=/path/to/data/
"""

import argparse
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Union

import gym
import h5py
import mani_skill2.envs
import numpy as np
import sapien.core as sapien
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import *
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.trajectory.merge_trajectory import merge_h5
from mani_skill2.utils.common import clip_and_scale_action, inv_scale_action
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.wrappers import RecordEpisode
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle


def qpos_to_pd_joint_delta_pos(controller: PDJointPosController, qpos):
    assert type(controller) == PDJointPosController
    assert controller.config.use_delta
    assert controller.config.normalize_action
    delta_qpos = qpos - controller.qpos
    low, high = controller.config.lower, controller.config.upper
    return inv_scale_action(delta_qpos, low, high)


def qpos_to_pd_joint_target_delta_pos(controller: PDJointPosController, qpos):
    assert type(controller) == PDJointPosController
    assert controller.config.use_delta
    assert controller.config.use_target
    assert controller.config.normalize_action
    delta_qpos = qpos - controller._target_qpos
    low, high = controller.config.lower, controller.config.upper
    return inv_scale_action(delta_qpos, low, high)


def qpos_to_pd_joint_vel(controller: PDJointVelController, qpos):
    assert type(controller) == PDJointVelController
    assert controller.config.normalize_action
    delta_qpos = qpos - controller.qpos
    qvel = delta_qpos * controller._control_freq
    low, high = controller.config.lower, controller.config.upper
    return inv_scale_action(qvel, low, high)


def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta


def delta_pose_to_pd_ee_delta(
    controller: Union[PDEEPoseController, PDEEPosController],
    delta_pose: sapien.Pose,
    pos_only=False,
):
    assert isinstance(controller, PDEEPosController)
    # assert controller.config.use_delta
    # assert controller.config.normalize_action
    low, high = controller._action_space.low, controller._action_space.high
    if pos_only:
        return inv_scale_action(delta_pose.p, low, high)
    delta_pose = np.r_[
        delta_pose.p,
        compact_axis_angle_from_quaternion(delta_pose.q),
    ]
    return inv_scale_action(delta_pose, low, high)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument(
        "-o",
        "--obs-mode",
        type=str,
        help="target observation mode",
        default="state_dict",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--save-traj", action="store_true", help="whether to save trajectories"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--max-retry", type=int, default=0)
    parser.add_argument(
        "--discard-timeout",
        action="store_true",
        help="whether to discard timeout episodes",
    )
    parser.add_argument(
        "--allow-failure", action="store_true", help="whether to allow failure episodes"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--use-env-states",
        action="store_true",
        help="whether to replay by env states instead of actions",
    )
    parser.add_argument(
        "--bg-name",
        type=str,
        default=None,
        help="background scene to use",
    )
    parser.add_argument(
        "--num-trajs",
        type=int,
        default=-1,
        help="number of trajectory to process",
    )
    return parser.parse_args()


def _main(args, proc_id: int = 0, num_procs=1, pbar=None):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = env_info["env_kwargs"]
    env_kwargs.update(obs_mode=args.obs_mode)

    env = gym.make(env_id, **env_kwargs)

    # Create a main env for replay
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )

    episodes = json_data["episodes"]
    n_ep = len(episodes)
    if args.num_trajs < n_ep and args.num_trajs > 0:
        n_ep = args.num_trajs
    inds = np.arange(n_ep)
    inds = np.array_split(inds, num_procs)[proc_id]

    # Replay
    for ind in inds:
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"]
        else:
            reset_kwargs["seed"] = ep["episode_seed"]

        ori_control_mode = ep["control_mode"]
        ori_actions = ori_h5_file[traj_id]["actions"][:]
        ori_states = ori_h5_file[traj_id]["obs"]  # ['extra']['tcp_pose'][:]
        # ori_states = ori_h5_file[traj_id]["obs"]['agent']['ee_pose'][:]

        env.reset(**reset_kwargs)
        obs = env.get_obs()
        for t in range(len(ori_actions)):
            # print("\n", 1, obs['agent']['ee_pose'], "\n", 2, obs['extra']['tcp_pose'])
            # obs = obs['extra']['tcp_pose']
            # obs = obs['agent']['ee_pose']

            if args.vis:
                env.render()

            curr_ee_pose_at_base = sapien.Pose(
                p=obs["extra"]["tcp_pose"][:3], q=obs["extra"]["tcp_pose"][3:]
            )
            # curr_ee_pose_at_base_2 = sapien.Pose(p=obs['agent']['ee_pose'][:3], q=obs['agent']['ee_pose'][3:])
            next_ee_pose_at_base = sapien.Pose(
                p=ori_states["extra"]["tcp_pose"][t + 1][:3],
                q=ori_states["extra"]["tcp_pose"][t + 1][3:],
            )
            # next_ee_pose_at_base_2 = sapien.Pose(p=ori_states['agent']['ee_pose'][t+1][:3], q=ori_states['agent']['ee_pose'][t+1][3:])

            ee_pose_at_ee = (
                curr_ee_pose_at_base.inv() * next_ee_pose_at_base
            )  # Pose (pos, quat)
            # ee_pose_at_ee_2 = curr_ee_pose_at_base_2.inv() * next_ee_pose_at_base_2 # Pose (pos, quat)
            arm_action = delta_pose_to_pd_ee_delta(
                env.agent.controller.controllers["arm"], ee_pose_at_ee, pos_only=False
            )  # Pose (pos, axis-angle)
            # arm_action_2 = delta_pose_to_pd_ee_delta(
            #     env.agent.controller.controllers["arm"], ee_pose_at_ee_2, pos_only=False
            # ) # Pose (pos, axis-angle)

            # print('\n', 1, arm_action, '\n', 2, arm_action_2, 3, ori_actions[t])

            if (np.abs(arm_action[:3])).max() > 1:  # position clipping
                arm_action[:3] = np.clip(arm_action[:3], -1, 1)
                flag = False
            arm_action = np.concatenate([arm_action, ori_actions[t][-1:]])
            print("\n", 1, arm_action, "\n", 2, ori_actions[t])

            obs, reward, done, info = env.step(arm_action)

            success = info.get("success", False)
            timeout = "TimeLimit.truncated" in info
            success = success and (not timeout)
            if success:
                print(tqdm.write(f"Episode {episode_id} replayed successfully"))

            # if success or args.allow_failure:
            #     env.flush_trajectory()
            #     env.flush_video()
            #     break
        # else:
        #     tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    # Cleanup
    if ori_env is not None:
        ori_env.close()
    ori_h5_file.close()

    if pbar is not None:
        pbar.close()

    return


def main():
    args = parse_args()

    if args.num_procs > 1:
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, args.num_procs) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
    else:
        _main(args)


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main()
