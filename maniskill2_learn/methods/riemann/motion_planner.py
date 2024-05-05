from typing import List, Tuple
import time

import torch
import numpy as np
from icecream import ic
from sapien.core import Pose
from pydrake.all import RigidTransform, PiecewisePolynomial, PiecewisePose
from transforms3d.quaternions import mat2quat

from maniskill2_learn.utils.torch import BaseAgent
from maniskill2_learn.methods.riemann.solver_utils import (
    anchor_seeds,
    solve_ik_joint,
    rotAxis,
    se3_inverse,
    dense_sample_traj_times,
    solve_ik_traj_with_standoff,
    build_plant,
    vector2pose,
    recursive_squeeze,
)
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import apply_pose_to_points

from ..builder import BRL
from .se3_pose_predictor import SE3PosePredictor


@BRL.register_module()
class RiemannAgent(BaseAgent):
    def __init__(
        self,
        env_params,
        diff_model: BaseAgent = None,
        actuation_time: int = 50,
        pre_actuation_times: List[int] = [40],
        post_actuation_times: List[int] = [65],
        pre_actuation_standby_time: int = 10,
        pre_actuation_motions: List[Tuple[str, int]] = [["translate_x", -0.02]],
        post_actuation_motions: List[Tuple[str, int]] = [["translate_x", 0.15]],
        **kwargs,
    ):
        super().__init__()
        self.env_params = env_params
        self.diff_model = diff_model
        self.pose_predictor = SE3PosePredictor()
        self.plant, self.fk_context = build_plant()
        self.reset_expert()

        self.actuation_time = actuation_time
        self.pre_actuation_times = pre_actuation_times
        self.post_actuation_times = post_actuation_times
        self.pre_actuation_motions = pre_actuation_motions
        self.post_actuation_motions = post_actuation_motions
        self.pre_actuation_standby_time = pre_actuation_standby_time

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
        translation = Pose(translation)
        actuation_pose = pre_pose * translation
        return actuation_pose

    def solve_postactuation_traj(self):
        """
        generate the full task trajectory with a FirstOrderHold
        """
        self.generate_actuation_poses()

    def generate_actuation_poses(self):
        self.pre_actuation_poses = []
        self.post_actuation_poses = []

        curr_pose = self.task_goal_hand_pose_in_base
        for motion in self.pre_actuation_motions:
            mode = motion[0]
            value = motion[1]

            assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
            assert type(value) is float

            if mode == "rotate":
                raise NotImplementedError
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

        curr_pose = self.task_goal_hand_pose_in_base
        for motion in self.post_actuation_motions:
            mode = motion[0]
            value = motion[1]

            assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
            assert type(value) is float or type(value) is int

            if mode == "rotate":
                raise NotImplementedError
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
            [0]
            + self.pre_actuation_times
            # + [self.actuation_time]
            + self.post_actuation_times
        )

        self.traj_keyframes = (
            [self.hand_pose_in_base]
            + self.pre_actuation_poses
            # + [self.task_goal_hand_pose_in_base]
            + self.post_actuation_poses
        )
        self.traj_keyframes = [
            pose.to_transformation_matrix() for pose in self.traj_keyframes
        ]

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

    def get_env_info(self, obs):
        self.dt = obs["dt"]  # simulator dt
        self.time = obs["time"].item()  # current time
        self.base_pose = vector2pose(obs["base_pose"])
        self.joint_positions = obs["joint_positions"]
        self.hand_pose = vector2pose(obs["hand_pose"])
        self.hand_pose_in_base = self.base_pose.inv() * self.hand_pose

        peg_pose = vector2pose(obs["peg_pose"])
        peg_head_offset = vector2pose(obs["peg_head_offset"])
        self.peg_head_pose = peg_pose.transform(peg_head_offset)
        box_hole_pose = vector2pose(obs["box_hole_pose"])
        self.goal_pose = box_hole_pose * peg_head_offset.inv()

    def solve_actuation_joint(self, pointcloud_obs):
        input_xyz = apply_pose_to_points(pointcloud_obs["xyz"], self.base_pose)
        input_rgb = pointcloud_obs["rgb"]

        input_xyz = torch.tensor(input_xyz).float().unsqueeze(0).to(self.device)
        input_rgb = torch.tensor(input_rgb).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            (
                pred_target_pos,
                pred_target_direction,
                pred_object_pos,
                pred_object_direction,
            ) = self.pose_predictor(input_xyz, input_rgb)

        pred_target_pos = pred_target_pos.squeeze(0).detach().cpu().numpy()
        pred_target_direction = pred_target_direction.squeeze(0).detach().cpu().numpy()
        pred_object_pos = pred_object_pos.squeeze(0).detach().cpu().numpy()
        pred_object_direction = pred_object_direction.squeeze(0).detach().cpu().numpy()

        target_pose = Pose(p=pred_target_pos, q=mat2quat(pred_target_direction))
        object_pose = Pose(p=pred_object_pos, q=mat2quat(pred_object_direction))

        hand_rel_pose = object_pose.inv() * self.hand_pose
        target_goal_hand_pose = target_pose * hand_rel_pose
        task_goal_hand_pose_in_base = self.base_pose.inv() * target_goal_hand_pose

        # DEBUG
        ic(object_pose.p, object_pose.q)
        ic(self.peg_head_pose.p, self.peg_head_pose.q)
        ic(target_pose.p, target_pose.q)
        ic(self.goal_pose.p, self.goal_pose.q)

        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        random_seeds = [self.joint_positions.copy()] + [
            anchor_seeds[idx] for idx in indexes
        ]
        solutions = []

        for seed in random_seeds:
            res = solve_ik_joint(
                gripper_frame=self.plant.GetFrameByName("panda_hand"),
                p_target=RigidTransform(task_goal_hand_pose_in_base.to_transformation_matrix()),
                q0=seed.reshape(-1, 1),
                centering_joint=self.joint_positions.copy(),
                consider_collision=False,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        if len(solutions) == 0:
            print("empty joint solution")
            self.goal_joint = self.joint_positions[:9].copy()
            self.solve_success = False
        else:
            self.solve_success = True
            solutions = np.array(solutions)
            joint_positions = self.joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(
                solutions - joint_positions.copy(), axis=-1
            )
            res = solutions[np.argmin(dist_to_init_joints)]
            self.goal_joint = res

            self.plant.SetPositions(self.fk_context, res)

        self.task_goal_hand_pose_in_base = self.plant.EvalBodyPoseInWorld(
            self.fk_context, self.plant.GetBodyByName("panda_hand")
        )
        self.task_goal_hand_pose_in_base = Pose.from_transformation_matrix(
            np.array(self.task_goal_hand_pose_in_base.GetAsMatrix4())
        )

    def solve_joint_traj(self, densify: bool = True):
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """

        keyposes = self.traj_keyframes
        keytimes = self.sample_times

        self.joint_traj_waypoints = [self.joint_positions.copy()]
        self.joint_space_traj = PiecewisePolynomial.FirstOrderHold(
            [0, self.actuation_time],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
        )

        if densify:
            self.dense_traj_times = dense_sample_traj_times(self.sample_times)
        else:
            self.dense_traj_times = self.sample_times

        print("solve traj endpoint")

        # interpolated joint
        res = solve_ik_traj_with_standoff(
            [
                self.hand_pose_in_base.to_transformation_matrix(),
                self.task_goal_hand_pose_in_base.to_transformation_matrix()
            ],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
            endpoint_times=[0, self.actuation_time],
            q_traj=self.joint_space_traj,
            waypoint_times=self.dense_traj_times,
            keytimes=keytimes,
            keyposes=keyposes,
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
                    self.joint_traj_waypoints[: self.pre_actuation_times[0]]
                    + [self.joint_traj_waypoints[self.pre_actuation_times[0]]]
                    * self.pre_actuation_standby_time
                    + self.joint_traj_waypoints[self.pre_actuation_times[0] :]
                )
                self.dense_traj_times = (
                    self.dense_traj_times[: self.pre_actuation_times[0]]
                    + [
                        self.dense_traj_times[self.pre_actuation_times[0]] + i
                        for i in range(self.pre_actuation_standby_time)
                    ]
                    + [
                        t + self.pre_actuation_standby_time
                        for t in self.dense_traj_times[self.pre_actuation_times[0] :]
                    ]
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
            # if densify:
            #     self.get_task_traj_from_joint_traj()
        else:
            print("endpoint trajectory not solved! environment termination")
            self.joint_plan_success = False
            # self.env.need_termination = True
            # if densify:
            #     self.get_task_traj_from_joint_traj()

    def forward(
        self, obs: dict, pointcloud_obs: dict = None, use_planner: bool = True, **kwargs
    ):
        if use_planner:
            obs = recursive_squeeze(obs, axis=0)
            if self.check_plan_empty() and pointcloud_obs is not None:
                pointcloud_obs = recursive_squeeze(pointcloud_obs, axis=0)
            self.get_env_info(obs)
            if self.check_plan_empty():
                s = time.time()
                self.solve_actuation_joint(pointcloud_obs)
                self.solve_postactuation_traj()
                self.solve_joint_traj()
                print("plan generation time: {:.3f}".format(time.time() - s))
                self.plan_time = self.time

            elapse_time = self.time - self.plan_time
            joint_action = self.joint_space_traj.value(elapse_time + self.dt).reshape(-1)
            maniskill_joint_action = np.concatenate(
                (joint_action[:7], -1 * np.ones_like(joint_action[:1])), axis=0
            )
            return maniskill_joint_action
        else:
            assert self.diff_model is not None
            return self.diff_model(obs, **kwargs)
