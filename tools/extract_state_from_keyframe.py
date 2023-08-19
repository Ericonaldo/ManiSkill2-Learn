import argparse
import os, numpy as np
import os.path as osp
from multiprocessing import Process
import h5py
import json

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


from maniskill2_learn.env import make_gym_env, ReplayMemory, import_env
from maniskill2_learn.utils.data import DictArray, GDict, f64_to_f32
from maniskill2_learn.utils.file import merge_h5_trajectory
from maniskill2_learn.utils.meta import get_total_memory, flush_print
from maniskill2_learn.utils.math import split_num

# from maniskill2_learn.utils.data import compress_f64

from transforms3d.quaternions import quat2axangle
import sapien.core as sapien

def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta

def auto_fix_wrong_name(traj):
    if isinstance(traj, GDict):
        traj = traj.memory
    for key in traj:
        if key in ["action", "reward", "done", "env_level", "next_env_level", "next_env_state", "env_state"]:
            traj[key + "s"] = traj[key]
            del traj[key]
    return traj


tmp_folder_in_docker = "/tmp"

def render(env):
    viewer = env.render("human")

def recursive_get_from_dict(dict_to_do, idx):
    res = {}
    for key in dict_to_do:
        if isinstance(dict_to_do[key], dict):
            res[key] = recursive_get_from_dict(dict_to_do[key], idx)
        else:
            res[key] = dict_to_do[key][idx]

    return res

def extract_keyframe_states(keys, args, worker_id, main_process_id):

    cnt = 0
    output_file = osp.join(tmp_folder_in_docker, f"{worker_id}.h5")
    output_h5 = h5py.File(output_file, "w")
    input_h5 = h5py.File(args.traj_name, "r")
    keyframe_h5 = h5py.File(args.keyframe_name, "r")

    max_keyframes_len = - 1
    for j, key in enumerate(keys):
        traj_keyframe = GDict.from_hdf5(keyframe_h5[key])["keyframe"]
        max_keyframes_len = max(len(traj_keyframe) - 1, max_keyframes_len)

    for j, key in enumerate(keys):
        trajectory = GDict.from_hdf5(input_h5[key])
        trajectory = auto_fix_wrong_name(trajectory)
        traj_keyframe = GDict.from_hdf5(keyframe_h5[key])["keyframe"]

        length = trajectory['actions'].shape[0]
        action_dim = trajectory['actions'].shape[-1]
        state_dim = trajectory['obs']['state'].shape[-1]

        replay = ReplayMemory(length)
        # action_accumulated = np.zeros(trajectory['actions'].shape)

        for i in range(length):
            # if i not in traj_keyframe:
            #     action_accumulated += trajectory['actions']
            #     continue
            
            # action_accumulated = trajectory['actions']
            item_i = recursive_get_from_dict(trajectory, i)
            # item_i["actions"] = action_accumulated
            difference_array = np.absolute(traj_keyframe-i)
            index = difference_array.argmin() # Find all keyframes after the frame i

            keyframe_action = np.zeros((max_keyframes_len,action_dim))
            keyframe_state = np.zeros((max_keyframes_len,state_dim))
            # keyframe_obs = {}
            # for key in trajectory['obs']:
            #     keyframe_obs[key] = np.zeros((max_keyframes_len,*trajectory['obs'][key].shape[1:]))
            keyframe_difference = np.zeros(max_keyframes_len)
            keyframe_mask = np.zeros(max_keyframes_len)
            if traj_keyframe[index] > i or (i == traj_keyframe[index] == length-1):
                first_key_frame = traj_keyframe[index]
                keyframe_difference[:traj_keyframe[index:].shape[0]] = traj_keyframe[index:]-i
                keyframe_action[:traj_keyframe[index:].shape[0]] = trajectory['actions'][traj_keyframe[index:]]
                keyframe_state[:traj_keyframe[index:].shape[0]] = trajectory['obs']['state'][traj_keyframe[index:]]
                # for key in trajectory['obs']:
                #     keyframe_obs[key][:traj_keyframe[index:].shape[0]] = trajectory['obs'][key][traj_keyframe[index:]]
                keyframe_mask[:traj_keyframe[index:].shape[0]] = 1.
                # keyframe_action = np.pad(traj_keyframe[index:], (0, max_keyframes_len-(len(traj_keyframe)-index)), 'constant', constant_values=(-1, -1))
            else:
                first_key_frame = traj_keyframe[index+1]
                keyframe_difference[:traj_keyframe[index+1:].shape[0]] = traj_keyframe[index+1:]-i
                keyframe_action[:traj_keyframe[index+1:].shape[0]] = trajectory['actions'][traj_keyframe[index+1:]]
                keyframe_state[:traj_keyframe[index+1:].shape[0]] = trajectory['obs']['state'][traj_keyframe[index+1:]]
                # for key in trajectory['obs']:
                #     keyframe_obs[key][:traj_keyframe[index+1:].shape[0]] = trajectory['obs'][key][traj_keyframe[index+1:]]
                keyframe_mask[:traj_keyframe[index+1:].shape[0]] = 1.
                # keyframe_action = np.pad(traj_keyframe[index+1:], (0, max_keyframes_len-(len(traj_keyframe)-index-1)), 'constant', constant_values=(-1, -1))
            item_i["keyframe_actions"] = keyframe_action
            item_i["keyframe_states"] = keyframe_state
            # item_i["keyframe_obs"] = keyframe_obs
            item_i["keyframe_masks"] = keyframe_mask
            item_i["keytime_differences"] = keyframe_difference
            item_i["timesteps"] = i
            item_i["ep_first_obs"] = recursive_get_from_dict(trajectory['obs'], 0)
            
            # We should not see the extra info about the goal, state dim 32
            # item_i["obs"]["state"] = item_i["obs"]["state"][...,:-6]

            # # Compute the angle
            cur_tcq_pose_np = item_i['obs']['state'][-7:]
            cur_tcq_pose = sapien.Pose(p=cur_tcq_pose_np[:3], q=cur_tcq_pose_np[3:])
            pose_np = np.r_[cur_tcq_pose.p, compact_axis_angle_from_quaternion(cur_tcq_pose.q)]
            item_i['obs']['state'] = np.concatenate([item_i['obs']['state'][:-7],pose_np], axis=-1)

            # We should not see the extra info about the goal, state dim 32
            # item_i["ep_first_obs"]["state"] = item_i["ep_first_obs"]["state"][...,:-6]

            # # Compute the angle
            cur_tcq_pose_np = item_i["ep_first_obs"]["state"][-7:]
            cur_tcq_pose = sapien.Pose(p=cur_tcq_pose_np[:3], q=cur_tcq_pose_np[3:])
            pose_np = np.r_[cur_tcq_pose.p, compact_axis_angle_from_quaternion(cur_tcq_pose.q)]
            item_i["ep_first_obs"]["state"] = np.concatenate([item_i["ep_first_obs"]["state"][:-7],pose_np], axis=-1)

            # We should not see the extra info about the goal, state dim 32
            # item_i["keyframe_states"] = item_i["keyframe_states"][...,:-6]

            # # Compute the angle
            tmp = []
            for k in range(len(item_i["keyframe_states"])):
                cur_tcq_pose_np = item_i["keyframe_states"][k,-7:]
                cur_tcq_pose = sapien.Pose(p=cur_tcq_pose_np[:3], q=cur_tcq_pose_np[3:])
                pose_np = np.r_[cur_tcq_pose.p, compact_axis_angle_from_quaternion(cur_tcq_pose.q)]
                tmp.append(np.concatenate([item_i["keyframe_states"][k,:-7],pose_np], axis=-1))
            item_i["keyframe_states"] = np.vstack(tmp)
            
            # print(item_i["obs"]["state"].shape, item_i["ep_first_obs"]["state"].shape, item_i["keyframe_states"].shape)

            assert first_key_frame >= i, "keyframe should after the frame, however {} < {}".format(item_i['keyframes'], i)

            item_i = GDict(item_i).f64_to_f32()
            replay.push(item_i)
            # action_accumulated = 0
            # print(item_i["obs"]['state'].shape)
            # print(item_i["keyframes"].shape)

        if worker_id == 0:
            flush_print(f"Extract keyframe trajectory: completed {cnt + 1} / {len(keys)}; this trajectory has length {length}")
        group = output_h5.create_group(f"traj_{cnt}")
        cnt += 1
        replay.to_hdf5(group, with_traj_index=False)
    output_h5.close()
    flush_print(f"Finish using {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract key frames")
    parser.add_argument("--env-id", help="Env name", type=str, default='PickCube')
    # Configurations
    parser.add_argument("--num-procs", default=1, type=int, help="Number of parallel processes to run, must be 1")
    parser.add_argument("--output-name", required=True, help="Output trajectory path, e.g. pickcube_pd_joint_delta_pos_pcd.h5")
    parser.add_argument("--traj-name", required=True, help="Input traj file name", type=str)
    parser.add_argument("--keyframe-name", required=True, help="Keyframe file name", type=str)
    parser.add_argument("--force", default=False, action="store_true", help="Force-regenerate the output trajectory file")
    parser.add_argument("--max-num-traj", default=-1, type=int, help="Maximum number of trajectories to convert from input file")

    args = parser.parse_args()
    args.traj_name = osp.abspath(args.traj_name)
    args.output_name = osp.abspath(args.output_name)
    return args

def main():
    os.makedirs(osp.dirname(args.output_name), exist_ok=True)
    if osp.exists(args.output_name) and not args.force:
        print(f"Trajectory generation for {args.env_name} with output path {args.output_name} has been completed!!")
        return

    with h5py.File(args.traj_name, "r") as h5_file:
        keys = sorted(h5_file.keys())
    if args.max_num_traj < 0:
        args.max_num_traj = len(keys)
    args.max_num_traj = min(len(keys), args.max_num_traj)
    args.num_procs = min(args.num_procs, args.max_num_traj)
    keys = keys[: args.max_num_traj]

    extra_args = ()

    if args.num_procs > 1:
        running_steps = split_num(len(keys), args.num_procs)[1]
        flush_print(f"Num of trajs = {len(keys)}", f"Num of process = {args.num_procs}")
        processes = []
        from copy import deepcopy

        for i, x in enumerate(running_steps):
            p = Process(target=extract_keyframe_states, args=(
                deepcopy(keys[:x]), args, i, os.getpid(), *extra_args))
            keys = keys[x:]
            processes.append(p)
            p.start()
        for p in processes:
            p.join()            
    else:
        running_steps = [len(keys)]
        extract_keyframe_states(keys, args, 0, os.getpid(), *extra_args)
    
    files = []
    for worker_id in range(len(running_steps)):
        tmp_h5 = osp.join(tmp_folder_in_docker, f"{worker_id}.h5")
        files.append(tmp_h5)
    from shutil import rmtree

    rmtree(args.output_name, ignore_errors=True)
    merge_h5_trajectory(files, args.output_name)
    for file in files:
        rmtree(file, ignore_errors=True)
    print(f"Finish merging files to {args.output_name}")


if __name__ == "__main__":
    args = parse_args()
    main()
