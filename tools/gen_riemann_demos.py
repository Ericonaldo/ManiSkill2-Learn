import numpy as np

from maniskill2_learn.env.env_utils import build_env, import_env
from maniskill2_learn.methods.kpam.kpam_utils import vector2pose
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import apply_pose_to_points


def run():
    traj_len = 20
    n_demos = 20
    n_points = 4096

    env_cfg = dict(
        type="gym",
        env_name="PegInsertionSide-v0",
        unwrapped=False,
        obs_mode="pointcloud",
        state_version="v2",
        history_len=6,
        control_mode="pd_joint_pos",
        n_points=n_points,
        camera_cfgs=dict(add_segmentation=True),
    )

    import_env()
    env = build_env(env_cfg)

    xyz_list, rgb_list, seg_center_list, axes_list = [], [], [], []
    for _ in range(n_demos):
        env.reset()
        obs = env.get_obs()
        kpam_obs = env.get_obs_kpam()
        goal_pose = env.goal_pose

        base_pose = vector2pose(kpam_obs["base_pose"])
        xyz = apply_pose_to_points(obs["xyz"], base_pose)
        rgb = obs["rgb"]
        seg_center = goal_pose.p
        axes = goal_pose.to_transformation_matrix()[:3, :3].reshape(-1)

        xyz = np.repeat(np.expand_dims(xyz, axis=0), traj_len, axis=0)
        rgb = np.repeat(np.expand_dims(rgb, axis=0), traj_len, axis=0)
        seg_center = np.repeat(np.expand_dims(seg_center, axis=0), traj_len, axis=0)
        axes = np.repeat(np.expand_dims(axes, axis=0), traj_len, axis=0)

        xyz_list.append(xyz)
        rgb_list.append(rgb)
        seg_center_list.append(seg_center)
        axes_list.append(axes)

    demo = {
        "xyz": np.stack(xyz_list, axis=0),
        "rgb": np.stack(rgb_list, axis=0),
        "seg_center": np.stack(seg_center_list, axis=0),
        "axes": np.stack(axes_list, axis=0),
    }
    np.savez(f"peginsert_demo_{n_points}.npz", **demo)


if __name__ == "__main__":
    run()
