python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" "agent_cfg.keyframe_pose_only=True" \
"workdir=posediff-quat-keydiff50000-seed$1" \
"agent_cfg.compatible=True" \
--evaluation --resume-from /NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_50000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" "agent_cfg.keyframe_pose_only=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-keydiff60000-seed$1" \
"agent_cfg.compatible=True" \
--evaluation --resume-from /NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_60000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" "agent_cfg.keyframe_pose_only=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-keydiff70000-seed$1" \
"agent_cfg.compatible=True" \
--evaluation --resume-from /NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_70000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" "agent_cfg.keyframe_pose_only=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-keydiff80000-seed$1" \
"agent_cfg.compatible=True" \
--evaluation --resume-from /NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_80000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" "agent_cfg.keyframe_pose_only=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-keydiff90000-seed$1" \
"agent_cfg.compatible=True" \
--evaluation --resume-from /NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_90000.ckpt &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" "agent_cfg.keyframe_pose_only=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-keydiff100000-seed$1" \
"agent_cfg.compatible=True" \
--evaluation --resume-from /NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_100000.ckpt
