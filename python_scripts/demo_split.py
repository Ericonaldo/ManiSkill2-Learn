import os
import argparse
from typing import List

import h5py
import numpy as np
from transforms3d.quaternions import quat2axangle

from maniskill2_learn.utils.data import GDict


def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta


def recursive_slice_from_dict(dict_to_do, start, end):
    res = {}
    for key in dict_to_do.keys():
        if isinstance(dict_to_do[key], dict):
            res[key] = recursive_slice_from_dict(dict_to_do[key], start, end)
        else:
            res[key] = dict_to_do[key][start:end]
    return res


def add_traj_rank(traj_item, rank_bins: int = 10):
    ee_traj = traj_item["obs"]["state"][:, 18 : 18 + 3]
    euc_dist = np.linalg.norm(ee_traj[-1] - ee_traj[0])
    dist = ee_traj[1:] - ee_traj[:-1]
    dist = np.linalg.norm(dist, axis=1).sum()

    rank = euc_dist / (dist + 1e-5)
    rank_id = rank / (1.0 / rank_bins)
    rank_one_hot = np.eye(rank_bins, dtype=np.float32)[int(np.clip(rank_id, 0, rank_bins - 1))]
    traj_item["rank"] = np.tile(rank_one_hot, (ee_traj.shape[0], 1))
    return traj_item


def _is_stopped(
    obs_traj,
    i,
    obs,
    stopped_buffer,
    delta: float = 0.01,
    grip_thresh: float = 0.03,
    use_gripper: bool = False,
):
    next_is_not_final = i == (len(obs_traj) - 2)
    if i <= 0 or (not use_gripper):
        gripper_state_no_change = True
    else:
        gripper_state_no_change = i < (len(obs_traj) - 2) and (
            obs[7] > grip_thresh == (obs_traj[i + 1][7] > grip_thresh)
            and obs[7] > grip_thresh == (obs_traj[i - 1][7] > grip_thresh)
            and (obs_traj[i - 2][7] > grip_thresh) == (obs_traj[i - 1][7] > grip_thresh)
        )
    joint_velocities = obs[9:18]
    small_delta = np.allclose(joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keyframe_detection_by_joints(
    demo,
    stopping_delta: float = 0.01,
    grip_thresh: float = 0.03,
    use_gripper: bool = False,
) -> List[int]:
    obs_traj = demo[0]
    episode_keypoints = []
    prev_gripper_open = obs_traj[0, 7] > grip_thresh
    stopped_buffer = 0
    for i, obs in enumerate(obs_traj):
        stopped = _is_stopped(
            obs_traj, i, obs, stopped_buffer, stopping_delta, grip_thresh
        )
        stopped_buffer = 15 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i == 0:
            gripper_open = True
        else:
            gripper_open = obs_traj[i][7] > grip_thresh

        if use_gripper:
            if i != 0 and (gripper_open != prev_gripper_open or last or stopped):
                episode_keypoints.append(i)
        elif i != 0 and (stopped or last):
            episode_keypoints.append(i)
        prev_gripper_open = gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def relabel_demo_action(
    traj_item, control_mode: str, rot_rep: str, grip_thresh: float = 0.03
):
    if control_mode == "pd_joint_pos":
        traj_item["actions"] = np.concatenate(
            [
                traj_item["obs"]["state"][:, :7],  # arm joint pos
                np.where(
                    traj_item["obs"]["state"][:, 8:9] > grip_thresh, 1.0, -1.0
                ),  # gripper open
            ],
            axis=-1,
        )
    elif control_mode == "pd_ee_pose":
        if rot_rep == "quat":
            traj_item["actions"] = np.concatenate(
                [
                    traj_item["obs"]["state"][:, 18 : 18 + 3],  # ee pos
                    # the maniskill pose controller take axis angle as action input
                    np.array(list(map(
                        compact_axis_angle_from_quaternion,
                        traj_item["obs"]["state"][:, 18 + 3 : 18 + 7],
                    ))),  # ee pose [pos, axis-angle]
                    np.where(
                        traj_item["obs"]["state"][:, 8:9] > grip_thresh, 1.0, -1.0
                    ),  # gripper open
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError(f"Unknown rotation representation: {rot_rep}")
    return traj_item


def parse_args():
    parser = argparse.ArgumentParser(description="Key frame identification")
    parser.add_argument("-f", "--filename", help="Replay file name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-test-split", action="store_true")
    parser.add_argument("--num-test-trajs", type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    filename = args.filename
    train_test_split = args.train_test_split
    num_test_trajs = args.num_test_trajs

    current_file = h5py.File(filename, "r")
    traj_keys = np.array(list(current_file.keys()))
    current_file.close()

    if train_test_split:
        np.random.seed(args.seed)
        np.random.shuffle(traj_keys)
        test_traj_keys, train_traj_keys = traj_keys[:num_test_trajs], traj_keys[num_test_trajs:]

        assert os.path.basename(filename).startswith("trajmslearn."), filename
        train_filename = os.path.join(
            os.path.dirname(filename),
            os.path.basename(filename).replace(".h5", ".train.h5"),
        )
        test_filename = os.path.join(
            os.path.dirname(filename),
            os.path.basename(filename).replace(".h5", ".test.h5"),
        )

        test_file = h5py.File(test_filename, "w")
        for key in test_traj_keys:
            group = test_file.create_group(key)
            item = GDict.from_hdf5(filename, keys=key)
            item.to_hdf5(group)
        test_file.close()

        train_file = h5py.File(train_filename, "w")
        for key in train_traj_keys:
            group = train_file.create_group(key)
            item = GDict.from_hdf5(filename, keys=key)
            item.to_hdf5(group)
        train_file.close()

        print(f"Save {len(test_traj_keys)} test trajectries to {test_filename}")
        print(f"Save {len(train_traj_keys)} train trajectries to {train_filename}")
