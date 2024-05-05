import pickle

import numpy as np

# from simple_3dviz.window import show
from open3d.geometry import OrientedBoundingBox
from open3d.utility import Vector3dVector
from simple_3dviz import Lines, Scene, Spherecloud
from simple_3dviz.utils import save_frame

from maniskill2_learn.methods.kpam.kpam_utils import vector2pose
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import apply_pose_to_points


def get_peg_pose_from_pointcloud(pointcloud_xyz, pointcloud_rgb, pointcloud_seg):

    def get_box_endpoints(box):
        box_points = np.asarray(box.get_box_points())
        start_point = box_points[0]
        dis_to_start_point = np.linalg.norm(box_points - start_point, axis=1)
        near_idx = np.argsort(dis_to_start_point)
        face_1_middle_point = (box_points[near_idx[1]] + box_points[near_idx[2]]) / 2

        start_point_2 = box_points[near_idx[-1]]
        dis_to_start_point_2 = np.linalg.norm(box_points - start_point_2, axis=1)
        near_idx_2 = np.argsort(dis_to_start_point_2)
        face_2_middle_point = (
            box_points[near_idx_2[1]] + box_points[near_idx_2[2]]
        ) / 2

        face_middle_points = np.stack(
            [face_1_middle_point, face_2_middle_point], axis=0
        )
        return face_middle_points

    # The 1st dimension is mesh-level (part) segmentation. The 2nd dimension is actor-level (object/link) segmentation.
    # [Actor(name="ground", id="16"), Actor(name="peg", id="17"), Actor(name="box_with_hole", id="18")]
    # head mesh id: 14, tail mesh id: 15
    peg_head_idx = np.where(
        (pointcloud_seg[:, 1] == 17) & (pointcloud_seg[:, 0] == 14)
    )[0]
    peg_head_pc = pointcloud_xyz[peg_head_idx]
    peg_head_o3d_vector = Vector3dVector(peg_head_pc)
    peg_head_bbox = OrientedBoundingBox.create_from_points(peg_head_o3d_vector)
    peg_head_middle_points = get_box_endpoints(peg_head_bbox)

    peg_tail_idx = np.where(
        (pointcloud_seg[:, 1] == 17) & (pointcloud_seg[:, 0] == 15)
    )[0]
    peg_tail_pc_center = np.mean(pointcloud_xyz[peg_tail_idx], axis=0)

    peg_head_middle_points = peg_head_middle_points[
        np.linalg.norm(peg_head_middle_points - peg_tail_pc_center, axis=-1).argsort()[
            ::-1
        ]
    ]
    return peg_head_middle_points  # head, middle


if __name__ == "__main__":
    with open("pointcloud.pkl", "rb") as f:
        pointcloud = pickle.load(f)

    # 1. get ground-truth information
    peg_pose = vector2pose(pointcloud["kpam_obs"]["peg_pose"])
    peg_head_offset = vector2pose(pointcloud["kpam_obs"]["peg_head_offset"])
    peg_head_pose = peg_pose.transform(peg_head_offset)

    # 2. transform base-frame pointclouds to the world frame
    base_pose = vector2pose(pointcloud["kpam_obs"]["base_pose"])
    pointcloud["xyz"] = apply_pose_to_points(pointcloud["xyz"], base_pose)

    # 3. create all-scene pointclouds object
    sizes = np.ones(pointcloud["xyz"].shape[0]) * 0.005
    pc = Spherecloud(pointcloud["xyz"], pointcloud["rgb"], sizes)

    # 4. extract position information from pointclouds
    peg_head_middle_points = get_peg_pose_from_pointcloud(
        pointcloud["xyz"], pointcloud["rgb"], pointcloud["seg"]
    )

    print(peg_head_middle_points[0])
    print(peg_head_pose.p)

    # show(pc, camera_position=(0.5, -0.5, 0.8), camera_target=(0.05, -0.1, 0.4))

    scene = Scene(size=(1024, 1024))
    scene.add(pc)
    # scene.add(bbox_lines)

    scene.camera_position = (0.5, -0.5, 0.8)
    scene.camera_target = (0.05, -0.1, 0.4)
    scene.up_vector = (0, 0, 1)

    # scene.camera_position = (-0.3, 0.15, 1.5)
    # scene.camera_target = (-0.3, 0.15, 0.4)
    # scene.up_vector = (1, 0, 0)

    scene.render()
    save_frame("pointcloud.png", scene.frame)
