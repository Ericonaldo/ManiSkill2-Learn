agent_cfg = dict(
    type="BC",
    batch_size=256,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="GaussianHead", init_log_std=-0.5, clip_return=True, predict_std=False
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(
                type="PointNet",
                feat_dim="pcd_all_channel",
                mlp_spec=[64, 128, 512],
                feature_transform=[],
            ),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
)

env_cfg = dict(
    type="gym",
    env_name="Fill-v0",
    unwrapped=False,
)


replay_cfg = dict(
    type="ReplayMemory",
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
    n_checkpoint=10000,
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
