import numpy as np
import argparse

from maniskill2_learn.env.env_utils import build_env, import_env
from maniskill2_learn.methods.kpam.kpam_utils import vector2pose
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import apply_pose_to_points


def run(args):
    env_cfg = dict(
        type="gym",
        env_name="PegInsertionSide-v0",
        unwrapped=False,
        obs_mode="pointcloud",
        state_version="v2",
        history_len=6,
        control_mode="pd_joint_pos",
        n_points=args.n_points,
        camera_cfgs=dict(add_segmentation=True),
        remove_arm_pointcloud=True,
        add_hole=not args.remove_hole,
        fixed_hole=args.fixed_hole,
        add_front_cover=args.add_front_cover,
    )

    import_env()
    env = build_env(env_cfg)
    env.seed(args.seed)

    xyz_list, rgb_list, seg_center_list, axes_list, object_seg_center_list, object_axes_list = [], [], [], [], [], []
    for _ in range(args.n_demos):
        env.reset()
        obs = env.get_obs()
        kpam_obs = env.get_obs_kpam()
        goal_pose = env.goal_pose
        peg_pose = vector2pose(kpam_obs["peg_pose"])
        peg_head_offset = vector2pose(kpam_obs["peg_head_offset"])
        peg_head_pose = peg_pose.transform(peg_head_offset)

        base_pose = vector2pose(kpam_obs["base_pose"])
        xyz = apply_pose_to_points(obs["xyz"], base_pose)
        rgb = obs["rgb"]
        seg_center = goal_pose.p
        # sapien use column major, so we need to transpose the rotation matrix
        axes = goal_pose.to_transformation_matrix()[:3, :3].T.reshape(-1)
        object_seg_center = peg_head_pose.p
        object_axes = peg_head_pose.to_transformation_matrix()[:3, :3].T.reshape(-1)

        xyz_list.append(xyz)
        rgb_list.append(rgb)
        seg_center_list.append(seg_center)
        axes_list.append(axes)
        object_seg_center_list.append(object_seg_center)
        object_axes_list.append(object_axes)

    demo = {
        "xyz": np.stack(xyz_list, axis=0),
        "rgb": np.stack(rgb_list, axis=0),
        "target_seg_center": np.stack(seg_center_list, axis=0),
        "target_axes": np.stack(axes_list, axis=0),
        "object_seg_center": np.stack(object_seg_center_list, axis=0),
        "object_axes": np.stack(object_axes_list, axis=0),
    }
    demo_file_name = f"peginsert_demo_{args.n_points}_s{args.seed}"
    if args.remove_hole:
        assert not args.fixed_hole
        demo_file_name += "_nohole"
    if args.fixed_hole:
        demo_file_name += "_fixhole"
    if args.add_front_cover:
        demo_file_name += "_frontcover"
    demo_file_name += ".npz"
    print(f"Saving demo npz to {demo_file_name}")
    np.savez(demo_file_name, **demo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n_demos", type=int, default=100)
    parser.add_argument("--n_points", type=int, default=4096)
    parser.add_argument("--remove_hole", action="store_true")
    parser.add_argument("--fixed_hole", action="store_true")
    parser.add_argument("--add_front_cover", action="store_true")
    args = parser.parse_args()
    run(args)
