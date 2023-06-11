horizon = 16
n_obs_steps = 8
future_action_len = horizon - n_obs_steps
eval_action_len = 2 # how many actions to be executed in the following timesteps for one input
workdir = "pcd"
agent_cfg = dict(
    type="DiffAgent",
    batch_size=128,
    action_seq_len=horizon,
    pcd_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
    visual_nn_cfg=dict(
        type="MultiImageObsEncoder", 
        shape_meta=dict(
            obs=dict(
                xyz=dict(
                    type="pcd",
                    shape="pcd_xyz_shape",
                ),
                rgb=dict(
                    type="pcd",
                    shape="pcd_rgb_shape",
                ),
                state=dict(
                    type="low_dim",
                    shape="agent_shape"
                ),
                frame_related_states=dict(
                    type="low_dim",
                    shape="pcd_frame_related_states_shape",
                ),
                to_frames=dict(
                    type="low_dim",
                    shape="pcd_to_frames_shape",
                ),
            )
        ),
        use_pcd_model=True,
    ),
    actor_cfg=dict(
        type="ContDiffActor",
    ),
    n_obs_steps=n_obs_steps, # n_obs_steps - 1 is the history length of the action, n_obs_steps is the history length of the observation
    obs_as_global_cond=True,
    fix_obs_stepd=True,
    action_visible=True,
    optim_cfg=dict(type="Adam", lr=3e-4),
    nn_cfg=dict(
        type="ConditionalUnet1D",
        input_dim="action_shape",
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ),
)

env_cfg = dict(
    type="gym",
    env_name="PickCube-v0",
    unwrapped=False,
    history_len=n_obs_steps,
    obs_mode="pointcloud",
    control_mode="pd_ee_delta_pose",
)


replay_cfg = dict(
    type="ReplayMemory",
    sampling_cfg=dict(
        type="TStepTransition",
        horizon=horizon,
    ),
    capacity=-1,
    num_samples=-1,
    keys=["obs", "actions", "dones", "episode_dones"],
    buffer_filenames=[
        "SOME_DEMO_FILE",
    ],
)

train_cfg = dict(
    on_policy=False,
    total_steps=50000,
    warm_steps=0,
    n_steps=0,
    n_updates=500,
    n_eval=50000,
    n_checkpoint=50000,
)

eval_cfg = dict(
    type="Evaluation",
    num=10,
    num_procs=1,
    use_hidden_state=False,
    save_traj=False,
    save_video=True,
    use_log=False,
)
