import os
import cv2
import csv
import numpy as np
import time
import peakutils
import argparse
import h5py
from tqdm import tqdm

# install from https://github.com/joelibaceta/video-keyframe-detector
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics
from maniskill2_learn.utils.file.cache_utils import *

def keyframeDetection(frames, dest, Thres, plotMetrics=False, verbose=False, suffix=""):
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Key frame identification")
    parser.add_argument("--env-id", help="Env name", type=str, default='PickCube')

    parser.add_argument("--replay-filenames", help="Replay file names", nargs='*')
    parser.add_argument("--threshold", help="Buffer file name", type=float, default=0.5)
    parser.add_argument("--control-mode", help="Control mode", type=str, default="pd_ee_delta_pose") # pd_ee_pose # pd_joint_pos
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    filenames = args.replay_filenames
    if filenames is None:
        filenames = [f"../ManiSkill2/demos/rigid_body/{args.env_id}-v0/trajmslearn.rgbd.{args.control_mode}.h5"]        

    for filename in filenames:
        res_dict ={}

        current_file = File(filename, "r")
        traj_keys = list(current_file.keys())
        current_file.close()
        
        # perspective = ["base"] # , "hand"]
        perspective = ["hand"]

        res_file = h5py.File(f"../ManiSkill2/demos/rigid_body/{args.env_id}-v0/keyframes.{args.control_mode}.h5", 'w')

        for key in tqdm(traj_keys):
            g = res_file.create_group(key)

            item = GDict.from_hdf5(filename, keys=key)
            item = DictArray(item)
            # print(item["obs"]["base_camera_rgbd"].shape)

            frame_idxes = [0, len(item["actions"])-1]

            for p in perspective:
                tmp_frames = item["obs"][f"{p}_camera_rgbd"][:,:3,:,:].astype(np.float32) # (C,H,W)
                tmp_frames = tmp_frames.swapaxes(1,2).swapaxes(2,3) # (H,W,C)
                tmp_frame_idxes = keyframeDetection(tmp_frames, f"key_frames/{args.env_id}", args.threshold, True, True, p)
                
                # print(p, tmp_frame_idxes)
                frame_idxes += list(tmp_frame_idxes)

            frame_idxes = list(set(frame_idxes))
            frame_idxes.sort()
            assert min(frame_idxes) >= 0, f"there are negative frame idxes!, {key}, {frame_idxes}"

            print(args.env_id, key, frame_idxes, len(frame_idxes))

            # res_dict[key] = np.array(frame_idxes).astype(np.int8)
            g.create_dataset('keyframe',data=np.array(frame_idxes).astype(int))
        
        res_file.close()
