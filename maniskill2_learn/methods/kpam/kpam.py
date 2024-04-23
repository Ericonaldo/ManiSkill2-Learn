import os
import time
import yaml

import numpy as np
from icecream import ic
from sapien.core import Pose
from pydrake.all import RigidTransform, PiecewisePolynomial, PiecewisePose
from open3d.geometry import OrientedBoundingBox
from open3d.utility import Vector3dVector
from transforms3d.quaternions import mat2quat
from mani_skill2.utils.sapien_utils import normalize_vector, vectorize_pose
from copy import deepcopy

from maniskill2_learn.utils.torch import BaseAgent
from maniskill2_learn.methods.kpam import se3_utils
from maniskill2_learn.methods.kpam.kpam_utils import (
    anchor_seeds,
    solve_ik_kpam,
    rotAxis,
    se3_inverse,
    dense_sample_traj_times,
    solve_ik_traj_with_standoff,
    build_plant,
    vector2pose,
    recursive_squeeze,
)
from maniskill2_learn.methods.kpam.optimization_spec import (
    OptimizationProblemSpecification,
)
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import apply_pose_to_points

from ..builder import BRL


@BRL.register_module()
class KPamAgent(BaseAgent):
    def __init__(self, env_params, diff_model=None, **kwargs):
        super().__init__()
        self.env_params = env_params
        self.diff_model = diff_model
        self.plant, self.fk_context = build_plant()
        self.reset_expert()
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg/peginsert.yaml")
        with open(cfg_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        self.actuation_time = self.cfg["actuation_time"]
        self.pre_actuation_times = self.cfg["pre_actuation_times"]
        self.post_actuation_times = self.cfg["post_actuation_times"]
        self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
        self.post_actuation_motions = self.cfg["post_actuation_motions"]
        self.pre_actuation_standby_time = self.cfg["pre_actuation_standby_time"]

    def reset(self, *args, **kwargs):
        self.reset_expert()

    def setup(self):
        # load keypoints
        self.solved_ik_times = []
        self.joint_traj_waypoints = []

    def reset_expert(self):
        """reinitialize expert state"""
        self.joint_space_traj = None
        self.task_space_traj = None
        self.plan_succeeded = False
        self.setup()

    def check_plan_empty(self):
        """check if already have a plan"""
        return self.joint_space_traj is None

    def get_pose_from_translation(self, translation, pre_pose):
        """get the pose from translation"""
        pose = np.eye(4)
        translation = np.array(translation)
        pose[:3, 3] = translation
        actuation_pose = pre_pose @ pose
        return actuation_pose

    def create_opt_problem(self, optimization_spec):
        """create a keypoint optimization problem from the current keypoint state"""
        optimization_spec.load_from_config(self.cfg)
        # print(f"load tool keypoint file from {self.cfg_path}")
        # # self.curr_tool_keypoint_head, self.curr_tool_keypoint_tail
        # # match the table
        # constraint_update_keypoint_target = {"tool_head": self.curr_object_keypoints[0]}

        # # minimize movement
        # cost_update_keypoint_target = {
        #     "tool_head": self.curr_tool_keypoints[0],
        #     "tool_tail": self.curr_tool_keypoints[1],
        #     "tool_side": self.curr_tool_keypoints[2],
        # }

        # for term in optimization_spec._cost_list:
        #     if hasattr(term, "keypoint_name") and term.keypoint_name in cost_update_keypoint_target.keys():
        #         term.target_position = cost_update_keypoint_target[term.keypoint_name]

        for term in optimization_spec._constraint_list:
            if (
                hasattr(term, "target_axis_frame")
                and term.target_axis_frame == "object"
            ):
                axis_inobject = term.target_axis.copy()
                term.target_axis = se3_utils.transform_vec(
                    self.object_pose, axis_inobject
                ).tolist()

        return optimization_spec

    # tool use related
    def solve_actuation_joint(self):
        """solve the formulated kpam problem and get goal joint"""

        optimization_spec = OptimizationProblemSpecification()
        optimization_spec = self.create_opt_problem(optimization_spec)

        constraint_dicts = [c.to_dict() for c in optimization_spec._constraint_list]

        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        random_seeds = [self.joint_positions.copy()] + [
            anchor_seeds[idx] for idx in indexes
        ]
        solutions = []

        for seed in random_seeds:
            res = solve_ik_kpam(
                constraint_dicts,
                self.plant.GetFrameByName("panda_hand"),
                self.tool_keypoints_in_hand,
                self.curr_object_keypoints,
                RigidTransform(self.ee_pose.reshape(4, 4)),
                seed.reshape(-1, 1),
                self.joint_positions.copy(),
                rot_tol=0.01,
                timeout=True,
                consider_collision=False,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        if len(solutions) == 0:
            print("empty solution in kpam")
            self.goal_joint = self.joint_positions[:9].copy()
            self.kpam_success = False
        else:
            self.kpam_success = True
            solutions = np.array(solutions)
            joint_positions = self.joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(
                solutions - joint_positions.copy(), axis=-1
            )
            res = solutions[np.argmin(dist_to_init_joints)]
            self.goal_joint = res

            self.plant.SetPositions(self.fk_context, res)

        self.task_goal_hand_pose = self.plant.EvalBodyPoseInWorld(
            self.fk_context, self.plant.GetBodyByName("panda_hand")
        )
        self.task_goal_hand_pose = np.array(self.task_goal_hand_pose.GetAsMatrix4())

    def solve_postactuation_traj(self):
        """
        generate the full task trajectory with a FirstOrderHold
        """
        self.generate_actuation_poses()

    def get_pose_from_rotation(self, rotation, pre_pose):
        """get the pose from rotation"""
        axis = self.env.get_object_axis()
        Rot = rotAxis(angle=rotation, axis=axis)
        actuation_pose = (
            self.object_pose @ Rot @ se3_inverse(self.object_pose) @ pre_pose
        )
        return actuation_pose

    def generate_actuation_poses(self):
        self.pre_actuation_poses = []
        self.post_actuation_poses = []

        curr_pose = self.task_goal_hand_pose
        for motion in self.pre_actuation_motions:
            mode = motion[0]
            value = motion[1]

            assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
            assert type(value) is float

            if mode == "rotate":
                curr_pose = self.get_pose_from_rotation(value, curr_pose)
                self.pre_actuation_poses.append(curr_pose)
            else:
                value_vec = [0, 0, 0]
                if mode == "translate_x":
                    value_vec[0] = value
                elif mode == "translate_y":
                    value_vec[1] = value
                elif mode == "translate_z":
                    value_vec[2] = value
                curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
                self.pre_actuation_poses.append(curr_pose)

        self.pre_actuation_poses.reverse()

        curr_pose = self.task_goal_hand_pose
        for motion in self.post_actuation_motions:
            mode = motion[0]
            value = motion[1]

            assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
            assert type(value) is float or type(value) is int

            if mode == "rotate":
                curr_pose = self.get_pose_from_rotation(value, curr_pose)
                self.post_actuation_poses.append(curr_pose)
            else:
                value_vec = [0, 0, 0]
                if mode == "translate_x":
                    value_vec[0] = value
                elif mode == "translate_y":
                    value_vec[1] = value
                elif mode == "translate_z":
                    value_vec[2] = value
                curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
                self.post_actuation_poses.append(curr_pose)

        self.sample_times = (
            [self.time]
            + [t + self.time for t in self.pre_actuation_times]
            # + [self.actuation_time + self.time]
            + [t + self.time for t in self.post_actuation_times]
        )

        self.traj_keyframes = (
            [self.ee_pose.reshape(4, 4)]
            + self.pre_actuation_poses
            # + [self.task_goal_hand_pose]
            + self.post_actuation_poses
        )

    def get_task_traj_from_joint_traj(self):
        """forward kinematics the joint trajectory to get the task trajectory"""
        self.pose_traj = []
        ik_times = dense_sample_traj_times(self.sample_times, self.actuation_time)
        self.dense_ik_times = ik_times
        for traj_time in ik_times:
            # diff_ik_context = self.differential_ik.GetMyMutableContextFromRoot(self.context)
            set_joints = self.joint_space_traj.value(traj_time)
            self.plant.SetPositions(self.fk_context, set_joints)
            pose = self.plant.EvalBodyPoseInWorld(
                self.fk_context, self.plant.GetBodyByName("panda_hand")
            )
            self.pose_traj.append(pose.GetAsMatrix4())

        self.task_space_traj = PiecewisePose.MakeLinear(
            ik_times, [RigidTransform(p) for p in self.pose_traj]
        )

    def solve_joint_traj(self, densify=True):
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """
        keyposes = self.traj_keyframes
        keytimes = self.sample_times

        self.joint_traj_waypoints = [self.joint_positions.copy()]
        self.joint_space_traj = PiecewisePolynomial.FirstOrderHold(
            [self.time, self.actuation_time + self.time],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
        )

        if densify:
            self.dense_traj_times = dense_sample_traj_times(self.sample_times)
        else:
            self.dense_traj_times = self.sample_times

        print("solve traj endpoint")

        # interpolated joint
        res = solve_ik_traj_with_standoff(
            [self.ee_pose.reshape(4, 4), self.task_goal_hand_pose],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
            endpoint_times=[self.time, self.actuation_time + self.time],
            q_traj=self.joint_space_traj,
            waypoint_times=self.dense_traj_times,
            keyposes=keyposes,
            keytimes=keytimes,
        )

        # solve the standoff and the remaining pose use the goal as seed.
        # stitch the trajectory
        if res is not None:
            # use the joint trajectory to build task trajectory for panda
            self.joint_plan_success = True
            self.joint_traj_waypoints = res.get_x_val().reshape(-1, 9)
            self.joint_traj_waypoints = list(self.joint_traj_waypoints)
            if self.pre_actuation_standby_time > 0:
                self.joint_traj_waypoints = (
                    self.joint_traj_waypoints[:self.pre_actuation_times[0]]
                    + [self.joint_traj_waypoints[self.pre_actuation_times[0]]] * self.pre_actuation_standby_time
                    + self.joint_traj_waypoints[self.pre_actuation_times[0]:]
                )
                self.dense_traj_times = (
                    self.dense_traj_times[:self.pre_actuation_times[0]]
                    + [self.dense_traj_times[self.pre_actuation_times[0]] + i for i in range(self.pre_actuation_standby_time)]
                    + [t + self.pre_actuation_standby_time for t in self.dense_traj_times[self.pre_actuation_times[0]:]]
                )

                # self.joint_traj_waypoints = (
                #     self.joint_traj_waypoints[:self.actuation_time]
                #     + [self.joint_traj_waypoints[self.actuation_time]] * self.pre_actuation_standby_time
                #     + self.joint_traj_waypoints[self.actuation_time:]
                # )
                # self.dense_traj_times = (
                #     self.dense_traj_times[:self.actuation_time]
                #     + [self.dense_traj_times[self.actuation_time] + i for i in range(self.pre_actuation_standby_time)]
                #     + [t + self.pre_actuation_standby_time for t in self.dense_traj_times[self.actuation_time:]]
                # )

            self.joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
                self.dense_traj_times, np.array(self.joint_traj_waypoints).T
            )
            if densify:
                self.get_task_traj_from_joint_traj()
        else:
            print("endpoint trajectory not solved! environment termination")
            self.joint_plan_success = False
            # self.env.need_termination = True
            if densify:
                self.get_task_traj_from_joint_traj()

    def compute_tool_keypoints_inbase(self):
        inv_base_pose = se3_inverse(self.base_pose)
        tool_keypoints_inbase = {}
        for name, loc in self.tool_keypoints_in_world.items():
            tool_keypoints_inbase[name] = inv_base_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        return tool_keypoints_inbase

    def compute_object_keypoints_inbase(self):
        inv_base_pose = se3_inverse(self.base_pose)
        object_keypoints_inbase = {}
        for name, loc in self.object_keypoints_in_world.items():
            object_keypoints_inbase[name] = inv_base_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        return object_keypoints_inbase

    def compute_hand_pose_inbase(self):
        ee_pose = self.ee_pose_in_world
        inv_base_pose = se3_inverse(self.base_pose)
        hand_pose_inbase = inv_base_pose.dot(ee_pose)
        hand_pose_inbase = hand_pose_inbase.reshape(4, 4)
        return hand_pose_inbase

    def compute_tool_pose_inbase(self):
        tool_pose = self.tool_pose_in_world
        inv_base_pose = se3_inverse(self.base_pose)
        tool_pose_inbase = inv_base_pose.dot(tool_pose)
        tool_pose_inbase = tool_pose_inbase.reshape(4, 4)
        return tool_pose_inbase

    def compute_object_pose_inbase(self):
        object_pose = self.object_pose_in_world
        inv_base_pose = se3_inverse(self.base_pose)
        object_pose_inbase = inv_base_pose.dot(object_pose)
        object_pose_inbase = object_pose_inbase.reshape(4, 4)
        return object_pose_inbase

    def compute_tool_keypoints_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        tool_keypoints_inhand = {}
        for name, loc in self.curr_tool_keypoints.items():
            tool_keypoints_inhand[name] = inv_ee_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        return tool_keypoints_inhand

    def compute_object_keypoints_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        object_keypoints_inhand = {}
        for name, loc in self.curr_object_keypoints.items():
            object_keypoints_inhand[name] = inv_ee_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        return object_keypoints_inhand

    def compute_tool_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        tool_rel_pose = inv_ee_pose.dot(self.tool_pose)
        return tool_rel_pose

    def get_env_info(self, obs):
        # get current end effector pose, joint angles, object poses, and keypoints from the environment
        self.tool_keypoints_in_world = {}
        peg_pose = vector2pose(obs["peg_pose"])
        peg_head_offset = vector2pose(obs["peg_head_offset"])
        peg_head_pose = peg_pose.transform(peg_head_offset)
        peg_tail_pose = peg_pose.transform(peg_head_offset.inv())
        peg_radius = obs["peg_half_size"][2]
        peg_side_pose = peg_head_pose.transform(Pose([0, 0, peg_radius]))
        self.tool_keypoints_in_world["tool_side"] = peg_side_pose.p
        self.tool_keypoints_in_world["tool_head"] = peg_head_pose.p
        self.tool_keypoints_in_world["tool_tail"] = peg_tail_pose.p

        self.object_keypoints_in_world = {}
        box_hole_pose = vector2pose(obs["box_hole_pose"])
        self.object_keypoints_in_world["object_head"] = (
            box_hole_pose * peg_head_offset.inv()
        ).p
        self.object_keypoints_in_world["object_tail"] = (
            box_hole_pose * peg_head_offset
        ).p

        self.dt = obs["dt"]  # simulator dt
        self.time = obs["time"].item()  # current time
        self.base_pose = vector2pose(obs["base_pose"]).to_transformation_matrix()
        self.joint_positions = obs["joint_positions"]

        self.ee_pose_in_world = vector2pose(obs["hand_pose"]).to_transformation_matrix()
        self.tool_pose_in_world = peg_pose.to_transformation_matrix()
        self.object_pose_in_world = (
            box_hole_pose * peg_head_offset.inv()
        ).to_transformation_matrix()

        self.curr_tool_keypoints = self.compute_tool_keypoints_inbase()
        self.curr_object_keypoints = self.compute_object_keypoints_inbase()
        self.ee_pose = self.compute_hand_pose_inbase()
        self.tool_pose = self.compute_tool_pose_inbase()
        self.object_pose = self.compute_object_pose_inbase()
        self.tool_keypoints_in_hand = self.compute_tool_keypoints_inhand()
        self.object_keypoints_in_hand = self.compute_object_keypoints_inhand()
        self.tool_rel_pose = self.compute_tool_inhand()

    def update_obs_with_pointcloud(self, obs, pointcloud_obs):
        obs = deepcopy(obs)
        base_pose = vector2pose(obs["base_pose"])
        pointcloud_obs["xyz"] = apply_pose_to_points(pointcloud_obs["xyz"], base_pose)

        def get_box_endpoints(box):
            box_points = np.asarray(box.get_box_points())
            start_point = box_points[0]
            dis_to_start_point = np.linalg.norm(box_points - start_point, axis=1)
            near_idx = np.argsort(dis_to_start_point)
            face_1_middle_point = (box_points[near_idx[1]] + box_points[near_idx[2]]) / 2

            start_point_2 = box_points[near_idx[-1]]
            dis_to_start_point_2 = np.linalg.norm(box_points - start_point_2, axis=1)
            near_idx_2 = np.argsort(dis_to_start_point_2)
            face_2_middle_point = (box_points[near_idx_2[1]] + box_points[near_idx_2[2]]) / 2

            face_middle_points = np.stack([face_1_middle_point, face_2_middle_point], axis=0)
            return face_middle_points

        # The 1st dimension is mesh-level (part) segmentation. The 2nd dimension is actor-level (object/link) segmentation.
        # [Actor(name="ground", id="16"), Actor(name="peg", id="17"), Actor(name="box_with_hole", id="18")]
        # head mesh id: 14, tail mesh id: 15
        peg_head_idx = np.where((pointcloud_obs["gt_seg"][:, 1] == 17) & (pointcloud_obs["gt_seg"][:, 0] == 14))[0]
        peg_head_pc = pointcloud_obs["xyz"][peg_head_idx]
        peg_head_o3d_vector = Vector3dVector(peg_head_pc)
        peg_head_bbox = OrientedBoundingBox.create_from_points(peg_head_o3d_vector)
        peg_positions = get_box_endpoints(peg_head_bbox)

        peg_tail_idx = np.where((pointcloud_obs["gt_seg"][:, 1] == 17) & (pointcloud_obs["gt_seg"][:, 0] == 15))[0]
        peg_tail_pc_center = np.mean(pointcloud_obs["xyz"][peg_tail_idx], axis=0)

        peg_head_position, peg_middle_position = peg_positions[np.linalg.norm(peg_positions - peg_tail_pc_center, axis=-1).argsort()[::-1]]

        def get_peg_head_pose(middle, head):
            forward = normalize_vector(head - middle)
            up = (0, 0, 1)
            left = np.cross(up, forward)
            forward = np.cross(left, up)  # use the fact that peg is lie flat on the table
            # up = np.cross(forward, left)
            rotation = np.stack([forward, left, up], axis=1)
            return Pose(p=head, q=mat2quat(rotation))

        # update obs
        peg_head_pose = get_peg_head_pose(peg_middle_position, peg_head_position)
        raw_peg_head_offset = vector2pose(obs["peg_head_offset"])
        peg_pose = peg_head_pose * raw_peg_head_offset.inv()
        obs["peg_pose"] = vectorize_pose(peg_pose)

        return obs

    def forward(self, obs, pointcloud_obs=None, use_kpam: bool = True, **kwargs):
        if use_kpam:
            obs = recursive_squeeze(obs, axis=0)
            if self.check_plan_empty() and pointcloud_obs is not None:
                pointcloud_obs = recursive_squeeze(pointcloud_obs, axis=0)
                obs = self.update_obs_with_pointcloud(obs, pointcloud_obs)
            self.get_env_info(obs)
            if self.check_plan_empty():
                s = time.time()
                self.solve_actuation_joint()
                self.solve_postactuation_traj()
                self.solve_joint_traj()
                print("plan generation time: {:.3f}".format(time.time() - s))
                self.plan_time = self.time
                # ic("plan", self.object_keypoints_in_hand["object_head"])
                # ic("plan", self.tool_keypoints_in_hand["tool_head"])

            if self.time == self.plan_time + self.pre_actuation_times[0] + self.pre_actuation_standby_time:
                ic("actuate", self.object_keypoints_in_hand["object_head"])
                ic("actuate", self.tool_keypoints_in_hand["tool_head"])

            joint_action = self.joint_space_traj.value(self.time + self.dt).reshape(-1)
            maniskill_joint_action = np.concatenate(
                (joint_action[:7], -1 * np.ones_like(joint_action[:1])), axis=0
            )
            return maniskill_joint_action
        else:
            assert self.diff_model is not None
            return self.diff_model(obs, **kwargs)
