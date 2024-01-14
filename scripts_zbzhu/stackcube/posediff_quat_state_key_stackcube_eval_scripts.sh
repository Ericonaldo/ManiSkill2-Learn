python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed 0 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$1-statekey$2-seed0" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230905_202043/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed 17898679 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$1-statekey$2-seed17898679" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230905_202043/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed 829384 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$1-statekey$2-seed829384" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230905_202043/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed 33794 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$1-statekey$2-seed33794" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230905_202043/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed 94 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$1-statekey$2-seed94" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230905_202043/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_poseonly_quat_rgbd_eval.py --seed 26272 \
--cfg-options "env_cfg.env_name=StackCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True"  \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=posediff-quat-diff$1-statekey$2-seed26272" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-statediff-rgbd-quat-nogradnorm-keylrschedule/20230905_202043/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/ManiSkill2/models/StackCube-posediff-statediff/newkeyframe-posediff-rgbd-quat-nogradnorm-keylrschedule/20230905_204240/models/model_$1.ckpt" &