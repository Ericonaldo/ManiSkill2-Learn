agent_cfg = dict(
    type="SAC",
    batch_size=256,  # Using multiple gpus leads to larger effective batch size, which can be crucial for SAC training
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
    shared_backbone=True,
    detach_actor_feature=True,
    alpha_optim_cfg=dict(type="Adam", lr=3e-4),
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhGaussianHead",
            log_std_bound=[-20, 2],
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
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape * 2"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4, param_cfg={"(.*?)visual_nn(.*?)": None}),
        # *Above removes visual_nn from actor optimizer; should only do so if shared_backbone=True and detach_actor_feature=True
        # *If either of the config options is False, then param_cfg={} should be removed, i.e. actor should also update the visual backbone.
        #   In addition, mlp_specs should be modified as well
        # *It is unknown if sharing backbone and detaching feature works well under the 3D setting. It is up to the users to figure this out.
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
)


train_cfg = dict(
    on_policy=False,
    total_steps=25000000,
    warm_steps=8000,
    n_eval=20000000,
    n_checkpoint=2000000,
    n_steps=32,
    n_updates=4,
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

env_cfg = dict(
    type="gym",
    env_name="PickCube-v0",
    obs_mode="pointcloud",
    ignore_dones=True,
)


replay_cfg = dict(
    type="ReplayMemory",
    capacity=400000,
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=4,
    with_info=True,
    multi_thread=False,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    num=10,
    use_hidden_state=False,
    save_traj=False,
    save_video=True,
    log_every_step=False,
    env_cfg=dict(ignore_dones=False),
)
