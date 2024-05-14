import os
import argparse
from typing import List

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from maniskill2_learn.env import ReplayMemory
from maniskill2_learn.utils.data import GDict, DictArray


def recursive_slice_from_dict(dict_to_do, start, end):
    res = {}
    for key in dict_to_do.keys():
        if isinstance(dict_to_do[key], dict):
            res[key] = recursive_slice_from_dict(dict_to_do[key], start, end)
        else:
            res[key] = dict_to_do[key][start:end]
    return res


def _is_stopped(
    obs_traj, i, obs, stopped_buffer, delta=0.01, grip_thresh=0.3, use_gripper=False
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


def keyframeDetectionByJoints(
    demo, stopping_delta=0.01, grip_thresh=0.4, use_gripper=False
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


def parse_args():
    parser = argparse.ArgumentParser(description="Key frame identification")
    parser.add_argument("-f", "--filename", help="Replay file name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    filename = args.filename

    current_file = h5py.File(filename, "r")
    traj_keys = list(current_file.keys())
    current_file.close()

    assert os.path.basename(filename).startswith("trajmslearn."), filename
    res_filename = os.path.join(
        os.path.dirname(filename),
        os.path.basename(filename).replace("trajmslearn.", "trajmslearn.keyframe."),
    )
    res_file = h5py.File(res_filename, "w")

    for key in traj_keys:
        item = GDict.from_hdf5(filename, keys=key)
        item = DictArray(item)

        keyframe_idxes = [0, len(item["actions"]) - 1]

        demos = (item["obs"]["state"], item["actions"])
        detect_frame_idxes = keyframeDetectionByJoints(demos)

        keyframe_idxes += list(detect_frame_idxes)
        keyframe_idxes = list(set(keyframe_idxes))
        keyframe_idxes.sort()

        if len(keyframe_idxes) >= 3:
            if keyframe_idxes[1] - keyframe_idxes[0] <= 6:
                keyframe_idxes.remove(keyframe_idxes[1])

        if len(keyframe_idxes) >= 3:
            if keyframe_idxes[-1] - keyframe_idxes[-2] <= 6:
                keyframe_idxes.remove(keyframe_idxes[-2])

        if len(keyframe_idxes) > 3 and "base_camera_rgbd" in item["obs"].keys():
            os.makedirs(f"./key_frames/base/{key}", exist_ok=True)
            os.makedirs(f"./key_frames/hand/{key}", exist_ok=True)
            for x in keyframe_idxes:
                cv2.imwrite(
                    os.path.join(f"./key_frames/base/{key}", str(x) + ".jpg"),
                    item["obs"]["base_camera_rgbd"][x, :3]
                    .swapaxes(0, 1)
                    .swapaxes(1, 2),
                )
                cv2.imwrite(
                    os.path.join(f"./key_frames/hand/{key}", str(x) + ".jpg"),
                    item["obs"]["hand_camera_rgbd"][x, :3]
                    .swapaxes(0, 1)
                    .swapaxes(1, 2),
                )

        print(f"Found {len(keyframe_idxes)} keypoints {keyframe_idxes} in {key}.")
        for i in range(len(keyframe_idxes) - 1):
            group = res_file.create_group(key + f"_{i}")
            start, end = keyframe_idxes[i : i + 2]
            replay = ReplayMemory(end - start)
            item_slice = GDict(recursive_slice_from_dict(item, start, end))
            replay.push_batch(item_slice)
            replay.to_hdf5(group, with_traj_index=False)

    res_file.close()
