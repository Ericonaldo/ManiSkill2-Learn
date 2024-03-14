python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$2-statekey80000-seed$1" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_80000.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230914_025228/models/model_$2.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$2-statekey120000-seed$1" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_120000.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230914_025228/models/model_$2.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$2-statekey150000-seed$1" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_150000.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230914_025228/models/model_$2.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed $1 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$2-statekey250000-seed$1" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230907_013059/models/model_250000.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/PegInsertionSide-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230914_025228/models/model_$2.ckpt" &
