workdir = "state_joint_pos_eval_first_range"
agent_cfg = dict(
    type="RiemannDiffAgent",
    diff_only=False,
    keyframe_modify_type="first_range",
    keyframe_modify_length=3,
)

horizon = 32
n_obs_steps = 6
future_action_len = horizon - n_obs_steps
eval_action_len = (
    27  # how many actions to be executed in the following timesteps for one input
)
diff_agent_cfg = dict(
    type="DiffAgent",
    batch_size=256,
    action_seq_len=horizon,
    n_obs_steps=n_obs_steps,  # n_obs_steps - 1 is the history length of the action, n_obs_steps is the history length of the observation
    obs_as_global_cond=True,
    fix_obs_stepd=True,
    action_visible=True,
    optim_cfg=dict(type="Adam", lr=3e-4),
    nn_cfg=dict(
        type="ConditionalUnet1D",
        input_dim="action_shape",
        local_cond_dim=None,
        global_cond_dim="agent_shape",
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ),
)

env_cfg = dict(
    type="gym",
    env_name="PegInsertionSide-v0",
    unwrapped=False,
    obs_mode="state",
    state_version="v2",
    history_len=n_obs_steps,
    control_mode="pd_joint_pos",
)
extra_env_cfg = dict(
    obs_mode="pointcloud",
    n_points=4096,
    camera_cfgs=dict(add_segmentation=True),
    remove_arm_pointcloud=True,
    add_front_cover=True,
)

eval_cfg = dict(
    type="RiemannEvaluation",
    num=100,
    num_procs=1,
    use_hidden_state=False,
    save_traj=False,
    save_video=False,
    use_log=False,
    eval_action_len=eval_action_len,
)
