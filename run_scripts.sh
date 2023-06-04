# Run PPO

python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py --g 0 
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=5" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5"

# "num_procs" controls parallelism during training. Details are described in later sections.

# FPS reported denotes the number of *control steps* per second.
# Note that the default simulation frequency in ManiSkill2 environments is 500hz, control frequency is 20hz.
# Therefore, 1 control step = 25 simulation steps.

# The above command does automatic evaluation after training. 
# Alternatively, you can manually evaluate a model checkpoint 
# by appending --evaluation and --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt 
# to the above commands.

# Run DAPG

python maniskill2_learn/apis/run_rl.py configs/mfrl/dapg/maniskill2_pn.py --g 0 \
--cfg-options "env_cfg.env_name=PegInsertionSide-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=5" \
"agent_cfg.demo_replay_cfg.buffer_filenames='./demos/rigid_body/PegInsertionSide-v0/trajectory.h5'" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5"

# To manually evaluate the model, 
# add --evaluation and --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt 
# to the above commands.

python maniskill2_learn/apis/run_rl.py configs/mfrl/dapg/maniskill2_pn.py --g 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=5" \
"agent_cfg.demo_replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajectory.pointcloud.pd_ee_delta_pose.h5'" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5"

# RUN BC

python maniskill2_learn/apis/run_rl.py configs/brl/bc/pointnet.py --g 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
"env_cfg.control_mode=pd_ee_delta_pose" \
"replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajectory.pointcloud.pd_ee_delta_pose.h5'" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"train_cfg.n_eval=50000" "train_cfg.total_steps=50000" "train_cfg.n_checkpoint=50000" "train_cfg.n_updates=500"


python maniskill2_learn/apis/run_rl.py configs/brl/bc/rgbd.py --g 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.n_points=1200" \
"env_cfg.control_mode=pd_ee_delta_pose" \
"replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"train_cfg.n_eval=50000" "train_cfg.total_steps=150000" "train_cfg.n_checkpoint=50000" "train_cfg.n_updates=500"



python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 4 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5" \
"train_cfg.n_eval=50000" "train_cfg.total_steps=500000" "train_cfg.n_checkpoint=50000" "train_cfg.n_updates=500"
# "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \



# Eval
python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd_eval.py --g 0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" \
"env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.num_procs=5" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
--evaluation --resume-from logs/DiffAgent/rgbd/20230604_033451/models/model_final.ckpt