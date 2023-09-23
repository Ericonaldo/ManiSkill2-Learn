# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_rgbd_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=posediff-angle-onlydiff50000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230914_025215/models/model_50000.ckpt &
# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_rgbd_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=posediff-angle-onlydiff60000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230914_025215/models/model_60000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-angle-onlydiff70000" \
"agent_cfg.compatible=True" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230914_025215/models/model_70000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-angle-onlydiff80000" \
"agent_cfg.compatible=True" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230914_025215/models/model_80000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-angle-onlydiff90000" \
"agent_cfg.compatible=True" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230914_025215/models/model_90000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_angle_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-angle-onlydiff100000" \
"agent_cfg.compatible=True" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-posediff-rgbd-angle-nogradnorm-keylrschedule/20230914_025215/models/model_100000.ckpt
