from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from icecream import ic
from pydrake.all import PiecewisePolynomial, RigidTransform
from sapien.core import Pose
from transforms3d.quaternions import mat2quat

from maniskill2_learn.methods.riemann.solver_utils import (
    anchor_seeds,
    build_plant,
    dense_sample_traj_times,
    recursive_squeeze,
    solve_ik_joint,
    solve_ik_traj_with_standoff,
    vector2pose,
)
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import apply_pose_to_points
from maniskill2_learn.utils.data import to_np
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.methods.riemann.se3_pose_predictor import SE3PosePredictor
from maniskill2_learn.utils.torch import BaseAgent

from ..builder import BRL


@BRL.register_module()
class RiemannDiffAgent(BaseAgent):
    def __init__(
        self,
        env_params,
        diff_model: BaseAgent,
        diff_only: bool = False,
        goal_threshold: float = 0.01,
        modify_time: int = 5,
        standby_time: int = 5,
        keyframe_modify_type: str = "last_range",
        keyframe_modify_length: int = 5,
        pre_actuation_motions: List[Tuple[str, int]] = [["translate_x", -0.02]],
        post_actuation_motions: List[Tuple[str, int]] = [["translate_x", 0.15]],
        pre_actuation_rel_times: List[int] = [-5],
        post_actuation_rel_times: List[int] = [15],
        **kwargs,
    ):
        super().__init__()
        self.env_params = env_params
        self.diff_model = diff_model
        self.plant, self.fk_context = build_plant()
        self.pose_predictor = SE3PosePredictor()

        self.diff_only = diff_only
        self.goal_threshold = goal_threshold
        self.modify_time = modify_time
        self.standby_time = standby_time
        self.pre_actuation_motions = pre_actuation_motions
        self.post_actuation_motions = post_actuation_motions
        self.pre_actuation_rel_times = pre_actuation_rel_times
        self.post_actuation_rel_times = post_actuation_rel_times

        assert keyframe_modify_type in [
            "all",
            "middle_range",
            "first_range",
            "last_range",
        ], keyframe_modify_type
        self.keyframe_modify_type = keyframe_modify_type
        self.keyframe_modify_length = keyframe_modify_length

        self._stage = "diffusion"

    @property
    def stage(self):
        return self._stage

    def kpam(self):
        if self.diff_only:
            return
        self._stage = "kpam"

    def reset(self, *args, **kwargs):
        self._stage = "diffusion"
        self.reset_expert()

    def setup(self):
        # load keypoints
        self.solved_ik_times = []
        self.joint_traj_waypoints = []

    def reset_expert(self):
        """reinitialize expert state"""
        self.joint_space_traj = None
        self.plan_succeeded = False
        self.target_pose = None
        self.goal_pose = None
        self.goal_hand_pose = None
        self.setup()

    def check_pose_empty(self):
        return (self.target_pose or self.goal_pose) is None

    def check_plan_empty(self):
        """check if already have a plan"""
        return self.joint_space_traj is None

    def predict_riemann_pose(self, dict_obs, pointcloud_obs):
        dict_obs = recursive_squeeze(dict_obs, axis=0)
        pointcloud_obs = recursive_squeeze(pointcloud_obs, axis=0)
        base_pose = vector2pose(dict_obs["base_pose"])

        input_xyz = apply_pose_to_points(pointcloud_obs["xyz"], base_pose)
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

        self.target_pose = Pose(p=pred_target_pos, q=mat2quat(pred_target_direction))
        self.object_pose = Pose(p=pred_object_pos, q=mat2quat(pred_object_direction))

    def get_goal_hand_pose(self, dict_obs):
        if self.goal_hand_pose is None:
            hand_pose = vector2pose(dict_obs["hand_pose"])
            hand_rel_pose = self.object_pose.inv() * hand_pose
            self.goal_hand_pose = self.target_pose * hand_rel_pose
        return self.goal_hand_pose

    def generate_actuation_poses(self, goal_hand_pose_in_base):
        pre_actuation_poses = []
        post_actuation_poses = []

        curr_pose = goal_hand_pose_in_base
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
                pre_actuation_poses.append(curr_pose)

        pre_actuation_poses.reverse()

        curr_pose = goal_hand_pose_in_base
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
                post_actuation_poses.append(curr_pose)

        return pre_actuation_poses, post_actuation_poses

    def forward(self, *args, **kwargs):
        return getattr(self, f"forward_with_{self.stage}")(*args, **kwargs)

    def forward_with_kpam(self, obs, dict_obs: Dict[str, np.ndarray], *args, **kwargs):
        dict_obs = recursive_squeeze(dict_obs, axis=0)
        if self.check_plan_empty():
            joint_positions = dict_obs["joint_positions"]
            modified_joint_positions, _ = self.modify_keyframe_joint(
                joint_positions, dict_obs
            )

            self.plant.SetPositions(self.fk_context, modified_joint_positions)
            goal_hand_pose_in_base = self.plant.EvalBodyPoseInWorld(
                self.fk_context, self.plant.GetBodyByName("panda_hand")
            )
            goal_hand_pose_in_base = np.array(goal_hand_pose_in_base.GetAsMatrix4())
            goal_hand_pose_in_base = Pose.from_transformation_matrix(goal_hand_pose_in_base)

            _, post_actuation_poses = self.generate_actuation_poses(goal_hand_pose_in_base)
            self.joint_space_traj, _ = self.solve_joint_traj(
                keyposes=[pose.to_transformation_matrix() for pose in post_actuation_poses],
                keytimes=[t + self.modify_time for t in self.post_actuation_rel_times],
                curr_joint_positions=joint_positions,
                goal_joint_positions=modified_joint_positions,
            )
            self.kpam_plan_time = dict_obs["time"].item()

        curr_time, dt = dict_obs["time"].item(), dict_obs["dt"]
        joint_action = self.joint_space_traj.value(
            curr_time - self.kpam_plan_time + dt
        ).reshape(-1)
        maniskill_joint_action = np.concatenate(
            (joint_action[:7], -1 * np.ones_like(joint_action[:1])), axis=0
        )
        return maniskill_joint_action

    def solve_joint_traj(
        self,
        keyposes,
        keytimes,
        curr_joint_positions,
        goal_joint_positions,
    ):
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """

        joint_traj_waypoints = [curr_joint_positions.copy()]
        joint_space_traj = PiecewisePolynomial.FirstOrderHold(
            [0.0, self.modify_time],
            np.array([curr_joint_positions.copy(), goal_joint_positions]).T,
        )

        dense_traj_times = dense_sample_traj_times([0, *keytimes])

        print("solve traj endpoint")

        # interpolated joint
        res = solve_ik_traj_with_standoff(
            np.array([curr_joint_positions.copy(), goal_joint_positions]).T,
            endpoint_times=[0, self.modify_time],
            q_traj=joint_space_traj,
            waypoint_times=dense_traj_times,
            keyposes=keyposes,
            keytimes=keytimes,
        )

        # solve the standoff and the remaining pose use the goal as seed.
        # stitch the trajectory
        if res is not None:
            # use the joint trajectory to build task trajectory for panda
            joint_plan_success = True
            joint_traj_waypoints = res.get_x_val().reshape(-1, 9)
            joint_traj_waypoints = list(joint_traj_waypoints)

            joint_traj_waypoints = (
                joint_traj_waypoints[: self.modify_time]
                + [joint_traj_waypoints[self.modify_time]] * self.standby_time
                + joint_traj_waypoints[self.modify_time :]
            )
            dense_traj_times = (
                dense_traj_times[: self.modify_time]
                + [
                    dense_traj_times[self.modify_time] + i
                    for i in range(self.standby_time)
                ]
                + [t + self.standby_time for t in dense_traj_times[self.modify_time :]]
            )

            joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
                dense_traj_times, np.array(joint_traj_waypoints).T
            )
        else:
            print("endpoint trajectory not solved! environment termination")
            joint_plan_success = False

        return joint_space_traj, joint_plan_success

    def get_pose_from_translation(self, translation, pre_pose):
        """get the pose from translation"""
        translation = Pose(translation)
        actuation_pose = pre_pose * translation
        return actuation_pose

    def forward_with_diffusion(
        self,
        observation: np.ndarray,
        dict_obs: Optional[Dict[str, np.ndarray]] = None,
        pointcloud_obs: Optional[Dict[str, np.ndarray]] = None,
        mode: str = "eval",
        *args,
        **kwargs,
    ):
        # if mode == "eval":  # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()

        observation = to_torch(observation, device=self.device, dtype=torch.float32)

        action_history = observation["actions"]
        if self.diff_model.obs_encoder is None:
            action_history = torch.cat(
                [action_history, torch.zeros_like(action_history[:, :1])],
                dim=1,
            )
            data = self.diff_model.normalizer.normalize(
                torch.cat((observation["state"], action_history), dim=-1)
            )
            observation["state"] = data[..., : observation["state"].shape[-1]]
            action_history = data[:, :-1, -self.diff_model.action_dim :]
        else:
            action_history = self.diff_model.normalizer.normalize(action_history)
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")

        self.diff_model.set_mode(mode=mode)

        act_mask, obs_mask = None, None
        if self.diff_model.fix_obs_steps:
            act_mask, obs_mask = self.diff_model.act_mask, self.diff_model.obs_mask

        if act_mask is None or obs_mask is None:
            if self.diff_model.obs_as_global_cond:
                act_mask, obs_mask, _ = self.diff_model.mask_generator(
                    (bs, self.diff_model.horizon, self.diff_model.action_dim), self.device
                )
                self.diff_model.act_mask, self.diff_model.obs_mask = act_mask, obs_mask
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        if act_mask.shape[0] < bs:
            act_mask = act_mask.repeat(max(bs // act_mask.shape[0] + 1, 2), 1, 1)
        if act_mask.shape[0] != bs:
            act_mask = act_mask[: action_history.shape[0]]  # obs mask is int

        if action_history.shape[1] == self.diff_model.horizon:
            for key in observation:
                observation[key] = observation[key][:, obs_mask, ...]

        if self.diff_model.obs_encoder is not None:
            obs_fea = self.diff_model.obs_encoder(
                observation
            )  # No need to mask out since the history is set as the desired length
        else:
            obs_fea = observation["state"].reshape(bs, -1)

        if self.diff_model.action_seq_len - hist_len:
            supp = torch.zeros(
                bs,
                self.diff_model.action_seq_len - hist_len,
                self.diff_model.action_dim,
                dtype=action_history.dtype,
                device=self.device,
            )
            action_history = torch.concat([action_history, supp], dim=1)

        pred_action_seq = self.diff_model.conditional_sample(
            cond_data=action_history,
            cond_mask=act_mask,
            global_cond=obs_fea,
            *args,
            **kwargs,
        )
        data = pred_action_seq
        if self.diff_model.obs_encoder is None:
            supp = torch.zeros(
                *pred_action_seq.shape[:-1],
                observation["state"].shape[-1],
                dtype=pred_action_seq.dtype,
                device=self.device,
            )
            data = torch.cat([supp, pred_action_seq], dim=-1)
        data = self.diff_model.normalizer.unnormalize(data)
        pred_action = data[..., -self.diff_model.action_dim :]

        if dict_obs is not None and not self.diff_only:
            dict_obs = recursive_squeeze(dict_obs, axis=0)
            self.get_goal_hand_pose(dict_obs)

            pred_is_keyframe = self.check_pred_keyframe(pred_action, dict_obs)
            keyframe_steps = np.where(pred_is_keyframe)[0]
            if len(keyframe_steps) > 0:
                if self.keyframe_modify_type == "middle_range":
                    if len(keyframe_steps) >= self.keyframe_modify_length:
                        middle_idx = len(keyframe_steps) // 2
                        half_length = self.keyframe_modify_length // 2
                        keyframe_steps = keyframe_steps[
                            middle_idx - half_length : middle_idx + half_length + 1
                        ]
                elif self.keyframe_modify_type == "first_range":
                    keyframe_steps = keyframe_steps[: self.keyframe_modify_length]
                elif self.keyframe_modify_type == "last_range":
                    keyframe_steps = keyframe_steps[self.keyframe_modify_length:]
                elif self.keyframe_modify_type == "all":
                    pass
                else:
                    raise ValueError(
                        f"Unknown keyframe_modify_type {self.keyframe_modify_type}"
                    )

                ic(keyframe_steps)

                modified_action_history = action_history.clone()
                modified_act_mask = act_mask.clone()
                for keyframe_step in keyframe_steps:
                    # transform to original joint format
                    pred_joint_positions = np.concatenate(
                        (
                            to_np(pred_action[:, keyframe_step])[0, :-1],
                            dict_obs["joint_positions"][-2:],
                        ),
                        axis=-1,
                    )
                    modified_joint_positions, solve_success = (
                        self.modify_keyframe_joint(pred_joint_positions, dict_obs)
                    )
                    modified_joint_positions = to_torch(
                        modified_joint_positions,
                        device=self.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    # transform back to maniskill format
                    modified_action = torch.cat(
                        (
                            modified_joint_positions[:, :-2],
                            pred_action[:, keyframe_step][:, -1:],
                        ),
                        dim=-1,
                    )

                    if solve_success:
                        supp = torch.zeros(
                            *modified_action.shape[:-1],
                            observation["state"].shape[-1],
                            dtype=modified_action.dtype,
                            device=self.device,
                        )
                        data = torch.cat([supp, modified_action], dim=-1)
                        data = self.diff_model.normalizer.normalize(data)
                        modified_action = data[..., -self.diff_model.action_dim :]

                        modified_action_history[:, keyframe_step] = modified_action
                        modified_act_mask[:, keyframe_step] = True

                pred_action_seq = self.diff_model.conditional_sample(
                    cond_data=modified_action_history,
                    cond_mask=modified_act_mask,
                    global_cond=obs_fea,
                    *args,
                    **kwargs,
                )
                data = pred_action_seq
                if self.diff_model.obs_encoder is None:
                    supp = torch.zeros(
                        *pred_action_seq.shape[:-1],
                        observation["state"].shape[-1],
                        dtype=pred_action_seq.dtype,
                        device=self.device,
                    )
                    data = torch.cat([supp, pred_action_seq], dim=-1)
                data = self.diff_model.normalizer.unnormalize(data)
                pred_action = data[..., -self.diff_model.action_dim :]

        if mode == "eval":
            pred_action = pred_action[:, -(self.diff_model.action_seq_len - hist_len) :]
            # Only used for ms-skill challenge online evaluation
            # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),-self.action_dim:]
            # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            #     for i in range(self.eval_action_len-1):
            #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,-self.action_dim:])

        return pred_action

    def is_preinserted(self, dict_obs):
        if self.goal_hand_pose is None:
            return False

        dict_obs = recursive_squeeze(dict_obs, axis=0)
        self.plant.SetPositions(self.fk_context, dict_obs["joint_positions"])
        hand_pose_inbase = self.plant.EvalBodyPoseInWorld(
            self.fk_context, self.plant.GetBodyByName("panda_hand")
        )
        hand_pose = vector2pose(
            dict_obs["base_pose"]
        ) * Pose.from_transformation_matrix(np.array(hand_pose_inbase.GetAsMatrix4()))
        hand_dis_to_goal = np.linalg.norm(hand_pose.p - self.goal_hand_pose.p)
        is_preinserted = hand_dis_to_goal < self.goal_threshold
        return is_preinserted

    def check_pred_keyframe(self, pred_action, dict_obs):
        res = [False for _ in range(self.diff_model.n_obs_steps - 1)]
        for step in range(self.diff_model.n_obs_steps - 1, pred_action.shape[1]):
            step_joint_pos = to_np(pred_action[0, step])
            drake_joint_pos = np.concatenate(
                [step_joint_pos[:-1], dict_obs["joint_positions"][-2:]], axis=-1
            )
            self.plant.SetPositions(self.fk_context, drake_joint_pos)
            hand_pose_inbase = self.plant.EvalBodyPoseInWorld(
                self.fk_context, self.plant.GetBodyByName("panda_hand")
            )
            hand_pose = vector2pose(
                dict_obs["base_pose"]
            ) * Pose.from_transformation_matrix(
                np.array(hand_pose_inbase.GetAsMatrix4())
            )
            hand_dis_to_goal = np.linalg.norm(hand_pose.p - self.goal_hand_pose.p)
            if hand_dis_to_goal < self.goal_threshold:
                is_keyframe = True
            else:
                is_keyframe = False
            res.append(is_keyframe)
        return np.array(res)

    def modify_keyframe_joint(self, joint_positions, dict_obs):
        assert not self.diff_only
        base_pose = vector2pose(dict_obs["base_pose"])
        task_goal_hand_pose_in_base = base_pose.inv() * self.goal_hand_pose

        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        random_seeds = [joint_positions.copy()] + [
            anchor_seeds[idx] for idx in indexes
        ]
        solutions = []

        for seed in random_seeds:
            res = solve_ik_joint(
                gripper_frame=self.plant.GetFrameByName("panda_hand"),
                p_target=RigidTransform(
                    task_goal_hand_pose_in_base.to_transformation_matrix()
                ),
                q0=seed.reshape(-1, 1),
                centering_joint=joint_positions.copy(),
                consider_collision=False,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        if len(solutions) == 0:
            print("empty joint solution")
            modified_joint = joint_positions[:9].copy()
            solve_success = False
        else:
            solve_success = True
            solutions = np.array(solutions)
            joint_positions = joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(
                solutions - joint_positions.copy(), axis=-1
            )
            res = solutions[np.argmin(dist_to_init_joints)]
            modified_joint = res

        return modified_joint, solve_success
