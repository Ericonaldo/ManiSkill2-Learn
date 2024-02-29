workdir = "mioe-rgbd"
agent_cfg = dict(
    type="BC",
    batch_size=256,
    actor_cfg=dict(
        type="ContinuousActor",
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(
                type="MultiImageObsEncoder",
                shape_meta=dict(
                    obs=dict(
                        base_camera_rgbd=dict(
                            type="rgbd", shape="image_size", channel=4
                        ),
                        hand_camera_rgbd=dict(
                            type="rgbd", shape="image_size", channel=4
                        ),
                        state=dict(type="low_dim", shape="agent_shape"),
                    )
                ),
                output_mlp=True,
                output_dim=256,
            ),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=[256, 128, "action_shape"],
                bias=True,
                inactivated_output=True,
            ),
        ),
        visual_dec_nn_cfg=dict(
            type="ImageObsDecoder",
            shape_meta=dict(
                shape_meta=dict(
                    obs=dict(
                        base_camera_rgbd=dict(
                            type="rgbd", shape="image_size", channel=4
                        ),
                        hand_camera_rgbd=dict(
                            type="rgbd", shape="image_size", channel=4
                        ),
                        state=dict(type="low_dim", shape="agent_shape"),
                    )
                ),
            ),
        ),
        head_cfg=dict(
            type="TanhHead",
            noise_std=1e-5,
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
)

# env_cfg = dict(
#     type="gym",
#     env_name="PickCube-v0",
#     unwrapped=False,
# )


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
    n_checkpoint=2000,
)

# eval_cfg = dict(
#     type="Evaluation",
#     num=10,
#     num_procs=1,
#     save_traj=False,
#     use_log=False,
# )
