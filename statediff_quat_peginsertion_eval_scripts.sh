python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-onlydiff50000" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_50000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-onlydiff60000" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_60000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-onlydiff70000" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_70000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-onlydiff80000" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_80000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-onlydiff90000" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_90000.ckpt &
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-onlydiff100000" \
--evaluation --resume-from ./logs/PegInsertionSide-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_100000.ckpt

