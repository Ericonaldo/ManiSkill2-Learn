python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff$1-key$2-seed0" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed 17898679 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff$1-key$2-seed17898679" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed 829384 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff$1-key$2-seed829384" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed 33794 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff$1-key$2-seed33794" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed 94 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff$1-key$2-seed94" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$1.ckpt" &

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/state_first_obs_rgbd_quat_targetpos_eval.py --seed 26272 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" "agent_cfg.use_keyframe=True" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=statediff-quat-key-diff$1-key$2-seed26272" \
"agent_cfg.compatible=True" \
"agent_cfg.keyframe_model_path=/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$2.ckpt" \
--evaluation --resume-from "/NAS2020/Datasets/DRLGroup/zbzhu/PickCube-v0/newkeyframe-statediff-rgbd-quat-tarpos-nogradnorm-keylrschedule/20230901_191222/models/model_$1.ckpt" &
