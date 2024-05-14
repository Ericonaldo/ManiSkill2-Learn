horizon = 64
n_obs_steps = 1
workdir = "state_v2_joint_pos"
agent_cfg = dict(
    type="KeyframeDiffAgent",
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

replay_cfg = dict(
    type="ReplayMemory",
    sampling_cfg=dict(
        type="ResampledTrajectory",
        horizon=horizon,
    ),
    capacity=-1,
    num_samples=-1,
    keys=["obs", "actions", "dones", "episode_dones"],
    buffer_filenames=[
        "SOME_DEMO_FILE",
    ],
    num_procs=4,
    synchronized=False,
)

train_cfg = dict(
    on_policy=False,
    total_steps=100000,
    warm_steps=0,
    n_steps=0,
    n_updates=500,
    n_eval=10000,
    n_checkpoint=10000,
)

eval_cfg = dict(
    type="OfflineDiffusionEvaluation",
    num=100,
    num_procs=1,
    use_hidden_state=False,
    save_traj=False,
    use_log=False,
)
