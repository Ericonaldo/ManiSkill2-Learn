# keyframe ee pose
python maniskill2_learn/apis/run_rl.py configs/keyframe_diff/state_pd_ee_pose.py --num-gpus 1 --env-id PlugCharger-v0 \
--cfg-options replay_cfg.buffer_filenames=./demos/rigid_body/PlugCharger-v0/pd_ee_pose/trajmslearn.keyframe.state.pd_ee_pose.train.h5 \
workdir=state_ee_pose_h48 horizon=48

# diff
python maniskill2_learn/apis/run_rl.py configs/brl/diff/state_pd_ee_pose.py --num-gpus 1 --env-id PlugCharger-v0 \
--cfg-options replay_cfg.buffer_filenames=./demos/rigid_body/PlugCharger-v0/pd_ee_pose/trajmslearn.state.pd_ee_pose.h5
