# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-quat-onlydiff70000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-quat-onlydiff100000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt 



# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-quat-key-diff70000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt &


# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-quat-key-diff100000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt &



# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-quat-key-diff70000-keyframe100000" \
# "agent_cfg.compatible=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt &


# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-quat-key-diff100000-keyframe70000" \
# "agent_cfg.compatible=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt &



python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff70000-keyframe80000" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_80000.ckpt" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff70000-keyframe120000" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_120000.ckpt" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff70000-keyframe160000" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_160000.ckpt" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_70000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff100000-keyframe80000" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_80000.ckpt" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff100000-keyframe120000" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_120000.ckpt" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff100000-keyframe160000" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_160000.ckpt" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_100000.ckpt &
