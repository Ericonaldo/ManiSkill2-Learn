
python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-angle-onlydiff50000" \
"agent_cfg.compatible=True" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_50000.ckpt &


python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-angle-onlydiff90000" \
"agent_cfg.compatible=True" \
--evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_90000.ckpt 



# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-onlydiff80000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt &


# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-onlydiff100000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt 






# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff80000-keyframe80000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff80000-keyframe100000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_120000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff80000-keyframe120000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_140000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff80000-keyframe140000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_150000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff80000-keyframe150000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt 



# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_80000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff100000-keyframe80000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff100000-keyframe100000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_120000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff100000-keyframe120000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_140000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff100000-keyframe140000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt &

# python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_angle_targetpos_eval.py --seed $1 \
# --cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
# "agent_cfg.keyframe_model_path=./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_150000.ckpt" \
# "eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
# "workdir=statediff-angle-key-diff100000-keyframe150000" \
# "agent_cfg.compatible=True" \
# --evaluation --resume-from ./logs/PickCube-v0/KeyDiffAgent/newkeyframe-statediff-rgbd-angle-tarpos-nogradnorm-keylrschedule/20230902_165900/models/model_100000.ckpt 
