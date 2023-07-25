n_obs_steps = 6
workdir = "bcrnn-rgbd"
agent_cfg = dict(
    type="BC",
    batch_size=128,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhHead",
            noise_std=1e-5,
            train_noise=False,
        ),
        nn_cfg=dict(
            type="RNNVisuomotor",
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
                output_dim=256,
            ),
            rnn_cfg=dict(
                type="SimpleRNN",
                input_dim=256,
                output_dim="action_shape",
                hidden_dim=256,
                n_layers=2,
                rnn_type="LSTM",
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    n_obs_steps=n_obs_steps,
)

# env_cfg = dict(
#     type="gym",
#     env_name="PickCube-v0",
#     unwrapped=False,
# )


replay_cfg = dict(
    type="ReplayMemory",
    sampling_cfg=dict(
        type="TStepTransition",
        horizon=n_obs_steps,
    ),
    capacity=-1,
    num_samples=-1,
    keys=["obs", "actions", "dones", "episode_dones"],
    buffer_filenames=[
        "SOME_DEMO_FILE",
    ],
    max_threads=4,
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

# eval_cfg = dict(
#     type="Evaluation",
#     num=10,
#     num_procs=1,
#     save_traj=False,
#     use_log=False,
# )
