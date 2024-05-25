python maniskill2_learn/apis/run_rl.py configs/keyframe_diff/state_pd_ee_pose.py --num-gpus 1 --env-id PegInsertionSide-v0 \
--cfg-options replay_cfg.buffer_filenames=./demos/rigid_body/PegInsertionSide-v0/pd_ee_pose/trajmslearn.keyframe.state_v2.pd_ee_pose.train.h5 \
workdir=state_v2_ee_pose_h48 horizon=48
