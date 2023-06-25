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


python maniskill2_learn/apis/run_rl.py configs/brl/bc/rgbd.py --num-gpus 8 --env-id PickCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"
# "env_cfg.control_mode=pd_ee_delta_pose" \

# RUN DiffPolicy

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PickCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_joint_pos.h5"
--auto-resume --work-dir="./logs/PickCube-v0/DiffAgent/rgbd-pd_joint_pos/20230618_162723"

python maniskill2_learn/apis/run_rl.py configs/brl/keyframe_diff/rgbd.py --num-gpus 8 --env-id PickCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.keyframes.rgbd.pd_joint_pos.h5"

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PickCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"
# "train_cfg.n_eval=50000" "train_cfg.total_steps=500000" "train_cfg.n_checkpoint=50000" "train_cfg.n_updates=500" \
# "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id StackCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/StackCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id AssemblingKits-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/AssemblingKits-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PlugCharger-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PlugCharger-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PegInsertionSide-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PegInsertionSide-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PlugCharger-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PlugCharger-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PandaAvoidObstacles-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PandaAvoidObstacles-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5" 

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PickClutterYCB-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickClutterYCB-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=4,5,6,7 python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd.py --num-gpus 8 --env-id PickSingleEGAD-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickSingleEGAD-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5" 

CUDA_VISIBLE_DEVICES=0,1,2,3 python ../void.py
## PCD

python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id PickCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"
# "train_cfg.n_eval=50000" "train_cfg.total_steps=500000" "train_cfg.n_checkpoint=50000" "train_cfg.n_updates=500" \
# "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" "env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
# "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \

CUDA_VISIBLE_DEVICES=1 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id StackCube-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/StackCube-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=2 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id AssemblingKits-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/AssemblingKits-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=3 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id PlugCharger-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PlugCharger-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=4 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id PegInsertionSide-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PegInsertionSide-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=5 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id PandaAvoidObstacles-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PandaAvoidObstacles-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=6 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id PickClutterYCB-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickClutterYCB-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

CUDA_VISIBLE_DEVICES=7 python maniskill2_learn/apis/run_rl.py configs/brl/diff/pcd.py --num-gpus 1 --env-id PickSingleEGAD-v0 \
--cfg-options "replay_cfg.buffer_filenames='./demos/rigid_body/PickSingleEGAD-v0/trajmslearn.pcd.pd_ee_delta_pose.h5"

# Eval
python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd_eval.py --g 0 --env-id PickCube-v0 \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" \
"env_cfg.control_mode=pd_ee_delta_pose" \
"eval_cfg.num=100" "eval_cfg.num_procs=5" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
--evaluation --resume-from logs/DiffAgent/rgbd/20230604_033451/models/model_final.ckpt

python maniskill2_learn/apis/run_rl.py configs/brl/diff/rgbd_eval.py --g 0 --build-replay \
--cfg-options "env_cfg.env_name=PickCube-v0" "env_cfg.obs_mode=rgbd" \
"env_cfg.control_mode=pd_ee_delta_pose" "replay_cfg.buffer_filenames='./demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5" \
"eval_cfg.num=100" "eval_cfg.num_procs=5" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
--evaluation --resume-from logs/DiffAgent/rgbd/20230606_021803/models/model_30000.ckpt
