# =============keyframes

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_relativekey_rgbd_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
# "workdir=posediff-angle-diff50000-relatkey$2" \
# "agent_cfg.keyframe_model_path=./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_$2.ckpt" \
# --evaluation --resume-from ./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_50000.ckpt &


# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_relativekey_rgbd_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
# "workdir=posediff-angle-diff60000-relatkey$2" \
# "agent_cfg.keyframe_model_path=./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_$2.ckpt" \
# --evaluation --resume-from ./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_60000.ckpt &



python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_relativekey_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
"workdir=posediff-angle-diff70000-relatkey$2" \
"agent_cfg.keyframe_model_path=./logs/StackCube-v0/KeyDiffAgent/newkeyframe-relativekeyframemodel-rgbd-angle-nogradnorm-keylrschedule/20230919_020452/models/model_$2.ckpt" \
--evaluation --resume-from ./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_70000.ckpt  &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_relativekey_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
"workdir=posediff-angle-diff80000-relatkey$2" \
"agent_cfg.keyframe_model_path=./logs/StackCube-v0/KeyDiffAgent/newkeyframe-relativekeyframemodel-rgbd-angle-nogradnorm-keylrschedule/20230919_020452/models/model_$2.ckpt" \
--evaluation --resume-from ./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_80000.ckpt  &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_relativekey_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
"workdir=posediff-angle-diff90000-relatkey$2" \
"agent_cfg.keyframe_model_path=./logs/StackCube-v0/KeyDiffAgent/newkeyframe-relativekeyframemodel-rgbd-angle-nogradnorm-keylrschedule/20230919_020452/models/model_$2.ckpt" \
--evaluation --resume-from ./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_90000.ckpt  &


python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_relativekey_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
"workdir=posediff-angle-diff100000-relatkey$2" \
"agent_cfg.keyframe_model_path=./logs/StackCube-v0/KeyDiffAgent/newkeyframe-relativekeyframemodel-rgbd-angle-nogradnorm-keylrschedule/20230919_020452/models/model_$2.ckpt" \
--evaluation --resume-from ./logs/StackCube-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230905_202127/models/model_100000.ckpt 
