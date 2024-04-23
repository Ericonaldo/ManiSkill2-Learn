SEED=$1

python maniskill2_learn/apis/run_kpam.py "configs/kpam/kpam_pointcloud_eval.py" --num-gpus 1 --seed ${SEED} \
--cfg-options "workdir=state_joint_pos_eval_kpam_pointcloud_rmrobot/seed_${SEED}" eval_cfg.save_video=False \
eval_cfg.num=100 eval_cfg.num_procs=1 env_cfg.env_name=PegInsertionSide-v0 env_cfg.state_version=v2 horizon=32 eval_action_len=27 n_obs_steps=6 \
--resume-from logs/PegInsertionSide-v0/DiffAgent/state_v2_joint_pos-obs_6-horizon_32/20240316_151348/models/model_80000.ckpt
