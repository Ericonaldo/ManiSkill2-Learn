MODIFY_TYPE=$1
SEED=$2

if [ $# -ne 2 ]; then
    echo "Need to provide modify type and seed"
    exit 1
fi

for MODIFY_LENGTH in 1 3 5
do
    python maniskill2_learn/apis/run_rl.py "configs/brl/kpam_diff/state_pd_joint_pos_${MODIFY_TYPE}_range_eval.py" --num-gpus 1 --seed ${SEED} \
    --cfg-options "agent_cfg.keyframe_modify_length=${MODIFY_LENGTH}" "workdir=state_joint_pos_eval_${MODIFY_TYPE}_range_${MODIFY_LENGTH}/seed_${SEED}" eval_cfg.save_video=False \
    eval_cfg.num=100 eval_cfg.num_procs=1 env_cfg.env_name=PegInsertionSide-v0 env_cfg.state_version=v2 horizon=32 eval_action_len=27 n_obs_steps=6 --evaluation \
    --resume-from logs/PegInsertionSide-v0/DiffAgent/state_v2_joint_pos-obs_6-horizon_32/20240316_151348/models/model_80000.ckpt
done