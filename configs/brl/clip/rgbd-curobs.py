horizon = 3
n_obs_steps = 2
future_action_len = horizon - n_obs_steps
workdir = "rgbd-curobs"
agent_cfg = dict(
    type="ClipAgent",
    batch_size=512,
    action_seq_len=horizon,
    current_obs_only=True,
    visual_nn_cfg=dict(
        type="MultiImageObsEncoder",
        shape_meta=dict(
            obs=dict(
                base_camera_rgbd=dict(
                    type="rgbd",
                    shape="image_size",
                    channel=4
                ),
                hand_camera_rgbd=dict(
                    type="rgbd",
                    shape="image_size",
                    channel=4
                ),
                state=dict(
                    type="low_dim",
                    shape="agent_shape"
                )
            )
        ),
        output_mlp=True,
        output_hidden_dims=[1024,512,256],
        output_dim=128,
    ),
    action_hidden_dims=[1024, 1024],
    temperature=1.0,
    n_obs_steps=n_obs_steps,  # n_obs_steps - 1 is the history length of the action, n_obs_steps is the history length of the observation
    obs_as_global_cond=True,
    fix_obs_stepd=True,
    action_visible=True,
    optim_cfg=dict(type="Adam", lr=3e-4),
)

# env_cfg = dict(
#     type="gym",
#     env_name="PickCube-v0",
#     unwrapped=False,
#     history_len=n_obs_steps,
#     obs_mode="rgbd",
#     control_mode="pd_ee_delta_pose"
# )

replay_cfg = dict(
    type="ReplayMemory",
    sampling_cfg=dict(
        type="TStepTransition",
        horizon=horizon,
        future_action_len=future_action_len,
    ),
    capacity=-1,
    num_samples=-1,
    keys=["obs", "actions", "dones", "episode_dones"],
    buffer_filenames=[
        "SOME_DEMO_FILE",
    ],
    num_procs=32,
    synchronized=False,
)

train_cfg = dict(
    on_policy=False,
    total_steps=500000,
    warm_steps=0,
    n_steps=0,
    n_updates=500,
    n_eval=10000,
    n_checkpoint=10000,
)

# eval_cfg = dict(
#     type="OfflineDiffusionEvaluation",
#     num=10,
#     num_procs=1,
#     use_hidden_state=False,
#     save_traj=False,
#     use_log=False,
# )
