# for epoch in 20000 40000 60000 80000 100000
# do
#     python maniskill2_learn/apis/run_rl.py configs/brl/diff/state_eval.py --seed 0 --num-gpus 1 \
#     --cfg-options "workdir=state_eval-obs_6-h_32-ep_$epoch" "eval_cfg.save_video=False" "eval_cfg.num=100" "eval_cfg.num_procs=20" "env_cfg.env_name=PegInsertionSide-v0" \
#     --evaluation --resume-from "logs/PegInsertionSide-v0/DiffAgent/state-obstep_6-horizon_32/20240312_154137/models/model_$epoch.ckpt"
# done


python maniskill2_learn/apis/run_rl.py configs/brl/diff/state_eval.py --seed 0 --num-gpus 1 \
--cfg-options "workdir=state_eval-obs_6-h_32-eval_27-ep_100000" "eval_cfg.save_video=False" "eval_cfg.num=100" "eval_cfg.num_procs=20" "env_cfg.env_name=PegInsertionSide-v0" \
"horizon=32" "eval_action_len=27" "n_obs_steps=6" "env_cfg.state_version=v1" \
--evaluation --resume-from "logs/PegInsertionSide-v0/DiffAgent/state_v1-obstep_6-horizon_32/20240315_065708/models/model_100000.ckpt"
