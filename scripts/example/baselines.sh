python maniskill2_learn/apis/run_rl.py configs/brl/bc/rgbd-mioe_eval.py --seed 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.type=Evaluation" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"workdir=bc-moie" \
--evaluation --resume-from ./logs/PickCube-v0/BC/bc-moie/20230712_033306/models/model_80000.ckpt