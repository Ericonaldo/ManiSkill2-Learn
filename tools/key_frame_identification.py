import os
import cv2
import csv
import numpy as np
import time
import peakutils
import argparse
import h5py
from tqdm import tqdm
from typing import List

# install from https://github.com/joelibaceta/video-keyframe-detector
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics
from maniskill2_learn.utils.file.cache_utils import *

def keyframeDetectionByFrames(frames, dest, Thres, plotMetrics=False, verbose=False, suffix=""):
    
    keyframePath = dest+f'/keyFrames_{suffix}'
    imageGridsPath = dest+f'/imageGrids_{suffix}'
    csvPath = dest+f'/csvFile_{suffix}'
    path2file = csvPath + f'/output_{suffix}.csv'
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    length = len(frames)

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    # Read until video is completed
    for i in range(length):
        frame = frames[i]
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        frame_number = i
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time-Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 12)
    indices = peakutils.indexes(y-base, Thres, min_dist=1)
    
    ##plot to monitor the selected keyframe
    if (plotMetrics):
        plot_metrics(indices, lstfrm, lstdiffMag, y, base, dest, suffix)

    cnt = 1
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
        cnt +=1
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if(verbose):
            print(log_message)
        with open(path2file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(log_message)
            csvFile.close()

    cv2.destroyAllWindows()

    return indices

# =========================
# Different keyframe detection methods
# =========================

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.01, grip_thresh=0.3, use_gripper=False):
    obss, acts = demo
    next_is_not_final = i == (len(obss) - 2)
    if i <= 0 or (not use_gripper):
        gripper_state_no_change = True
    else:
        gripper_open = acts[i-1][-1] > grip_thresh
        gripper_state_no_change = (
                i < (len(obss) - 2) and
                (gripper_open == (acts[i][-1] > grip_thresh) and
                gripper_open == (acts[i-2][-1] > grip_thresh) and
                (acts[i-3][-1] > grip_thresh) == (acts[i-2][-1] > grip_thresh)))
    joint_velocities = obs[9:18]
    small_delta = np.allclose(joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def keyframeDetectionByJoints(demo, stopping_delta=0.01, grip_thresh=0.4, use_gripper=False) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = True
    stopped_buffer = 0
    obss, acts = demo
    for i, obs in enumerate(obss):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta, grip_thresh)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i == 0:
            gripper_open = True
        else:
            gripper_open = acts[i-1][-1] > grip_thresh
        
        if use_gripper:
            if i != 0 and (gripper_open != prev_gripper_open or last or stopped):
                episode_keypoints.append(i)
        else:
            if i != 0 and stopped:
                episode_keypoints.append(i)
        prev_gripper_open = gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def parse_args():
    parser = argparse.ArgumentParser(description="Key frame identification")
    parser.add_argument("--env-id", help="Env name", type=str, default='PickCube')

    parser.add_argument("--replay-filenames", help="Replay file names", nargs='*')
    parser.add_argument("--threshold", help="Buffer file name", type=float, default=0.5)
    parser.add_argument("--body", help="Body", type=str, default="rigid") # soft
    parser.add_argument("--control-mode", help="Control mode", type=str, default="pd_ee_delta_pose") # pd_ee_pose # pd_joint_pos
    parser.add_argument("--keyframe-mode", help="Keyframe detection mode", type=str, default="joints") # video
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    filenames = args.replay_filenames
    if filenames is None:
        filenames = [f"../ManiSkill2/demos/{args.body}_body/{args.env_id}-v0/trajmslearn.rgbd.{args.control_mode}.h5"]        

    for filename in filenames:
        res_dict ={}

        current_file = File(filename, "r")
        traj_keys = list(current_file.keys())
        current_file.close()
        
        res_file = h5py.File(f"../ManiSkill2/demos/{args.body}_body/{args.env_id}-v0/keyframes.{args.control_mode}.h5", 'w')

        perspective = ["base"] # , "hand"]
        # perspective = ["hand"]

        for key in tqdm(traj_keys):
            g = res_file.create_group(key)

            item = GDict.from_hdf5(filename, keys=key)
            item = DictArray(item)
            # print(item["obs"]["base_camera_rgbd"].shape)

            frame_idxes = [0, len(item["actions"])-1]

            if args.keyframe_mode == "video":

                for p in perspective:
                    tmp_frames = item["obs"][f"{p}_camera_rgbd"][:,:3,:,:].astype(np.float32) # (C,H,W)
                    tmp_frames = tmp_frames.swapaxes(1,2).swapaxes(2,3) # (H,W,C)
                    tmp_frame_idxes = keyframeDetectionByFrames(tmp_frames, f"key_frames/{args.env_id}", args.threshold, True, True, p)
                    
                    # print(p, tmp_frame_idxes)
                    frame_idxes += list(tmp_frame_idxes)

            elif args.keyframe_mode == "joints":
                tmp_demos = (item["obs"]["state"], item["actions"])
                tmp_frame_idxes = keyframeDetectionByJoints(tmp_demos)
                cnt = 0
                frame_idxes += list(tmp_frame_idxes)
            
            else:
                raise NotImplementedError

            frame_idxes = list(set(frame_idxes))
            frame_idxes.sort()
            if len(frame_idxes) >= 3:
                if frame_idxes[1] - frame_idxes[0] <= 6:
                    frame_idxes.remove(frame_idxes[1])
            if len(frame_idxes) >= 3:
                if frame_idxes[-1] - frame_idxes[-2] <= 6:
                    frame_idxes.remove(frame_idxes[-2])
            if len(frame_idxes) > 3:
                os.makedirs(f"./key_frames/{args.env_id}/base/{key}", exist_ok=True)
                os.makedirs(f"./key_frames/{args.env_id}/hand/{key}", exist_ok=True)
                for x in frame_idxes:
                    cv2.imwrite(os.path.join(f"./key_frames/{args.env_id}/base/{key}", str(x) +'.jpg'), item["obs"]["base_camera_rgbd"][x,:3].swapaxes(0,1).swapaxes(1,2))
                    cv2.imwrite(os.path.join(f"./key_frames/{args.env_id}/hand/{key}", str(x) +'.jpg'), item["obs"]["hand_camera_rgbd"][x,:3].swapaxes(0,1).swapaxes(1,2))
            # exit(0)
            assert min(frame_idxes) >= 0, f"there are negative frame idxes!, {key}, {frame_idxes}"

            print(args.env_id, key, frame_idxes, len(frame_idxes))

            # res_dict[key] = np.array(frame_idxes).astype(np.int8)
            g.create_dataset('keyframe',data=np.array(frame_idxes).astype(int))
        
        res_file.close()
