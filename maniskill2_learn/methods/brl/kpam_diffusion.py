import os
import yaml
from typing import Dict, Optional
import torch
import numpy as np
from sapien.core import Pose
from pydrake.all import RigidTransform
from icecream import ic

from maniskill2_learn.methods.brl import DiffAgent
from maniskill2_learn.utils.torch import get_mean_lr
from maniskill2_learn.utils.diffusion.arrays import to_torch
from maniskill2_learn.utils.data import to_np
from maniskill2_learn.methods.kpam import se3_utils
from maniskill2_learn.methods.brl.kpam_diff_utils import (
    anchor_seeds,
    solve_ik_kpam,
    build_plant,
    vector2pose,
    recursive_squeeze,
)
from maniskill2_learn.methods.kpam.optimization_spec import (
    OptimizationProblemSpecification,
)

from ..builder import BRL


@BRL.register_module()
class KPamDiffAgent(DiffAgent):
    def __init__(
        self,
        keyframe_modify_type: str = "middle",
        keyframe_modify_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.plant, self.fk_context = build_plant()
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg/peginsert.yaml")
        with open(cfg_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        assert keyframe_modify_type in [
            "middle",
            "first",
            "last",
            "all",
            "middle_range",
            "first_range",
        ], keyframe_modify_type
        self.keyframe_modify_type = keyframe_modify_type
        if "range" in self.keyframe_modify_type:
            self.keyframe_modify_length = keyframe_modify_length

    def create_opt_problem(self, optimization_spec):
        """create a keypoint optimization problem from the current keypoint state"""
        optimization_spec.load_from_config(self.cfg)
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

    def forward(
        self,
        observation: np.ndarray,
        kpam_obs: Optional[Dict[str, np.ndarray]] = None,
        returns_rate: float = 0.9,
        mode: str = "eval",
        *args,
        **kwargs,
    ):
        # if mode == "eval":  # Only used for ms-skill challenge online evaluation
        #     if self.eval_action_queue is not None and len(self.eval_action_queue):
        #         return self.eval_action_queue.popleft()

        observation = to_torch(observation, device=self.device, dtype=torch.float32)

        action_history = observation["actions"]
        if self.obs_encoder is None:
            action_history = torch.cat(
                [action_history, torch.zeros_like(action_history[:, :1])],
                dim=1,
            )
            data = self.normalizer.normalize(
                torch.cat((observation["state"], action_history), dim=-1)
            )
            observation["state"] = data[..., : observation["state"].shape[-1]]
            action_history = data[:, :-1, -self.action_dim :]
        else:
            action_history = self.normalizer.normalize(action_history)
        bs = action_history.shape[0]
        hist_len = action_history.shape[1]
        observation.pop("actions")

        self.set_mode(mode=mode)

        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask

        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask, _ = self.mask_generator(
                    (bs, self.horizon, self.action_dim), self.device
                )
                self.act_mask, self.obs_mask = act_mask, obs_mask
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        if act_mask.shape[0] < bs:
            act_mask = act_mask.repeat(max(bs // act_mask.shape[0] + 1, 2), 1, 1)
        if act_mask.shape[0] != bs:
            act_mask = act_mask[: action_history.shape[0]]  # obs mask is int

        if action_history.shape[1] == self.horizon:
            for key in observation:
                observation[key] = observation[key][:, obs_mask, ...]

        if self.obs_encoder is not None:
            obs_fea = self.obs_encoder(
                observation
            )  # No need to mask out since the history is set as the desired length
        else:
            obs_fea = observation["state"].reshape(bs, -1)

        if self.action_seq_len - hist_len:
            supp = torch.zeros(
                bs,
                self.action_seq_len - hist_len,
                self.action_dim,
                dtype=action_history.dtype,
                device=self.device,
            )
            action_history = torch.concat([action_history, supp], dim=1)

        pred_action_seq = self.conditional_sample(
            cond_data=action_history,
            cond_mask=act_mask,
            global_cond=obs_fea,
            *args,
            **kwargs,
        )
        data = pred_action_seq
        if self.obs_encoder is None:
            supp = torch.zeros(
                *pred_action_seq.shape[:-1],
                observation["state"].shape[-1],
                dtype=pred_action_seq.dtype,
                device=self.device,
            )
            data = torch.cat([supp, pred_action_seq], dim=-1)
        data = self.normalizer.unnormalize(data)
        pred_action = data[..., -self.action_dim :]

        if kpam_obs is not None:
            kpam_obs = recursive_squeeze(kpam_obs, axis=0)
            pred_is_keyframe = self.check_pred_keyframe(pred_action, kpam_obs)
            keyframe_steps = np.where(pred_is_keyframe)[0]
            if len(keyframe_steps) > 0:
                if self.keyframe_modify_type == "first":
                    keyframe_steps = np.array([keyframe_steps[0]])
                elif self.keyframe_modify_type == "last":
                    keyframe_steps = np.array([keyframe_steps[-1]])
                elif self.keyframe_modify_type == "middle":
                    keyframe_steps = np.array(
                        [keyframe_steps[len(keyframe_steps) // 2]]
                    )
                elif self.keyframe_modify_type == "middle_range":
                    if len(keyframe_steps) >= self.keyframe_modify_length:
                        middle_idx = len(keyframe_steps) // 2
                        half_length = self.keyframe_modify_length // 2
                        keyframe_steps = keyframe_steps[
                            middle_idx - half_length : middle_idx + half_length + 1
                        ]
                        assert (
                            len(keyframe_steps) == self.keyframe_modify_length
                        ), f"{len(keyframe_steps)}, {self.keyframe_modify_length}"
                elif self.keyframe_modify_type == "first_range":
                    keyframe_steps = keyframe_steps[: self.keyframe_modify_length]
                elif self.keyframe_modify_type == "all":
                    pass
                else:
                    raise ValueError(
                        f"Unknown keyframe_modify_type {self.keyframe_modify_type}"
                    )

                kpam_action_history = action_history.clone()
                kpam_act_mask = act_mask.clone()
                for keyframe_step in keyframe_steps:
                    modified_action, solve_success = self.modify_keyframe_joint(
                        pred_action[:, keyframe_step], kpam_obs
                    )
                    if solve_success:
                        supp = torch.zeros(
                            *modified_action.shape[:-1],
                            observation["state"].shape[-1],
                            dtype=modified_action.dtype,
                            device=self.device,
                        )
                        data = torch.cat([supp, modified_action], dim=-1)
                        data = self.normalizer.normalize(data)
                        modified_action = data[..., -self.action_dim :]

                        kpam_action_history[:, keyframe_step] = modified_action
                        kpam_act_mask[:, keyframe_step] = True

                pred_action_seq = self.conditional_sample(
                    cond_data=kpam_action_history,
                    cond_mask=kpam_act_mask,
                    global_cond=obs_fea,
                    *args,
                    **kwargs,
                )
                data = pred_action_seq
                if self.obs_encoder is None:
                    supp = torch.zeros(
                        *pred_action_seq.shape[:-1],
                        observation["state"].shape[-1],
                        dtype=pred_action_seq.dtype,
                        device=self.device,
                    )
                    data = torch.cat([supp, pred_action_seq], dim=-1)
                data = self.normalizer.unnormalize(data)
                pred_action = data[..., -self.action_dim :]

        if mode == "eval":
            pred_action = pred_action[:, -(self.action_seq_len - hist_len) :]
            # Only used for ms-skill challenge online evaluation
            # pred_action = pred_action_seq[:,-(self.action_seq_len-hist_len),-self.action_dim:]
            # if (self.eval_action_queue is not None) and (len(self.eval_action_queue) == 0):
            #     for i in range(self.eval_action_len-1):
            #         self.eval_action_queue.append(pred_action_seq[:,-(self.action_seq_len-hist_len)+i+1,-self.action_dim:])

        return pred_action

    def check_pred_keyframe(self, pred_action, kpam_obs):
        res = [False for _ in range(self.n_obs_steps - 1)]
        peg_rel_pose = vector2pose(kpam_obs["hand_pose"]).inv() * vector2pose(
            kpam_obs["peg_pose"]
        )
        goal_pose = (
            vector2pose(kpam_obs["box_hole_pose"])
            * vector2pose(kpam_obs["peg_head_offset"]).inv()
        )
        for step in range(self.n_obs_steps - 1, pred_action.shape[1]):
            step_joint_pos = to_np(pred_action[0, step])
            drake_joint_pos = np.concatenate(
                [step_joint_pos[:-1], kpam_obs["joint_positions"][-2:]], axis=-1
            )
            self.plant.SetPositions(self.fk_context, drake_joint_pos)
            hand_pose_inbase = self.plant.EvalBodyPoseInWorld(
                self.fk_context, self.plant.GetBodyByName("panda_hand")
            )
            hand_pose = vector2pose(
                kpam_obs["base_pose"]
            ) * Pose.from_transformation_matrix(
                np.array(hand_pose_inbase.GetAsMatrix4())
            )
            peg_pose = hand_pose * peg_rel_pose
            peg_head_pose = peg_pose * vector2pose(kpam_obs["peg_head_offset"])

            peg_head_wrt_goal = goal_pose.inv() * peg_head_pose
            peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
            peg_head_wrt_goal_x_offset = peg_head_wrt_goal.p[0]
            peg_wrt_goal = goal_pose.inv() * peg_pose
            peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
            if (
                peg_head_wrt_goal_yz_dist < 0.01
                and peg_wrt_goal_yz_dist < 0.01
                and peg_head_wrt_goal_x_offset > -0.015
                and peg_head_wrt_goal_x_offset <= 0.0
            ):
                is_keyframe = True
            else:
                is_keyframe = False
            res.append(is_keyframe)
        return np.array(res)

    def modify_keyframe_joint(self, current_joint, kpam_obs, verify: bool = False):
        # Reduce batch dim and transform to raw format
        joint_positions = np.concatenate(
            (to_np(current_joint)[0, :-1], kpam_obs["joint_positions"][-2:]), axis=-1
        )
        self.plant.SetPositions(self.fk_context, joint_positions)
        hand_pose_inbase = Pose.from_transformation_matrix(
            np.array(
                self.plant.EvalBodyPoseInWorld(
                    self.fk_context, self.plant.GetBodyByName("panda_hand")
                ).GetAsMatrix4()
            )
        )
        base_pose = vector2pose(kpam_obs["base_pose"])
        hand_pose = base_pose * hand_pose_inbase

        tool_keypoints_in_hand = {}
        peg_rel_pose = vector2pose(kpam_obs["hand_pose"]).inv() * vector2pose(
            kpam_obs["peg_pose"]
        )
        peg_pose = hand_pose * peg_rel_pose
        peg_head_offset = vector2pose(kpam_obs["peg_head_offset"])
        peg_head_pose = peg_pose.transform(peg_head_offset)
        peg_tail_pose = peg_pose.transform(peg_head_offset.inv())
        peg_radius = kpam_obs["peg_half_size"][2]
        peg_side_pose = peg_head_pose.transform(Pose([0, 0, peg_radius]))
        tool_keypoints_in_hand["tool_side"] = (hand_pose.inv() * peg_side_pose).p
        tool_keypoints_in_hand["tool_head"] = (hand_pose.inv() * peg_head_pose).p
        tool_keypoints_in_hand["tool_tail"] = (hand_pose.inv() * peg_tail_pose).p

        goal_pose = (
            vector2pose(kpam_obs["box_hole_pose"])
            * vector2pose(kpam_obs["peg_head_offset"]).inv()
        )
        curr_peg_head_pose = (
            hand_pose * peg_rel_pose * vector2pose(kpam_obs["peg_head_offset"])
        )
        peg_head_wrt_goal = goal_pose.inv() * curr_peg_head_pose
        standoff_pose = Pose([peg_head_wrt_goal.p[0], 0.0, 0.0])

        curr_object_keypoints_in_base = {}
        box_hole_pose = vector2pose(kpam_obs["box_hole_pose"])
        curr_object_keypoints_in_base["object_head"] = (
            base_pose.inv() * (box_hole_pose * peg_head_offset.inv() * standoff_pose)
        ).p
        curr_object_keypoints_in_base["object_tail"] = (
            base_pose.inv() * (box_hole_pose * peg_head_offset)
        ).p

        if verify:
            tool_head_pos = peg_head_pose.p
            object_head_pos = (box_hole_pose * peg_head_offset.inv() * standoff_pose).p
            ic("before", tool_head_pos, object_head_pos)

        optimization_spec = OptimizationProblemSpecification()
        optimization_spec = self.create_opt_problem(optimization_spec)
        constraint_dicts = [c.to_dict() for c in optimization_spec._constraint_list]

        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        random_seeds = [joint_positions.copy()] + [anchor_seeds[idx] for idx in indexes]
        solutions = []

        for seed in random_seeds:
            res = solve_ik_kpam(
                constraint_dicts,
                self.plant.GetFrameByName("panda_hand"),
                tool_keypoints_in_hand,
                curr_object_keypoints_in_base,
                RigidTransform(hand_pose_inbase.to_transformation_matrix()),
                seed.reshape(-1, 1),
                joint_positions.copy(),
                rot_tol=0.01,
                timeout=True,
                consider_collision=False,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        if len(solutions) == 0:
            print("empty solution in kpam")
            modified_joint = joint_positions[:9].copy()
            kpam_success = False
        else:
            kpam_success = True
            solutions = np.array(solutions)
            joint_positions = joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(
                solutions - joint_positions.copy(), axis=-1
            )
            modified_joint = solutions[np.argmin(dist_to_init_joints)]

        if verify:
            self.plant.SetPositions(self.fk_context, modified_joint)
            hand_pose_inbase = Pose.from_transformation_matrix(
                np.array(
                    self.plant.EvalBodyPoseInWorld(
                        self.fk_context, self.plant.GetBodyByName("panda_hand")
                    ).GetAsMatrix4()
                )
            )
            base_pose = vector2pose(kpam_obs["base_pose"])
            hand_pose = base_pose * hand_pose_inbase
            peg_pose = hand_pose * peg_rel_pose
            peg_head_offset = vector2pose(kpam_obs["peg_head_offset"])
            tool_head_pos = (peg_pose.transform(peg_head_offset)).p
            object_head_pos = (box_hole_pose * peg_head_offset.inv() * standoff_pose).p
            ic("after", tool_head_pos, object_head_pos)

        modified_joint = to_torch(
            modified_joint, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        # Transform back to maniskill format
        modified_joint = torch.cat(
            (modified_joint[:, :-2], current_joint[:, -1:]), dim=-1
        )
        return modified_joint, kpam_success

    def update_parameters(self, memory, updates):
        if not self.init_normalizer:
            # Fit normalizer
            if self.obs_encoder is None:
                data = np.concatenate(
                    (memory.get_all("obs", "state"), memory.get_all("actions")), axis=-1
                )
            else:
                data = memory.get_all("actions")
            self.normalizer.fit(data, last_n_dims=1, mode="limits", range_eps=1e-7)
            self.init_normalizer = True

        batch_size = self.batch_size
        if self.obs_encoder is None:
            sample_kwargs = {"obsact_normalizer": self.normalizer}
        else:
            sample_kwargs = {"action_normalizer": self.normalizer}
        sampled_batch = memory.sample(
            batch_size,
            device=self.device,
            obs_mask=self.obs_mask,
            require_mask=True,
            **sample_kwargs,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.actor_optim.zero_grad()
        # {'obs': {'base_camera_rgbd': [(bs, horizon, 4, 128, 128)], 'hand_camera_rgbd': [(bs, horizon, 4, 128, 128)],
        # 'state': (bs, horizon, 38)}, 'actions': (bs, horizon, 7), 'dones': (bs, 1),
        # 'episode_dones': (bs, horizon, 1), 'worker_indices': (bs, 1), 'is_truncated': (bs, 1), 'is_valid': (bs, 1)}

        # generate impainting mask
        traj_data = sampled_batch["normed_actions"]
        masked_obs = sampled_batch["obs"]
        if self.obs_encoder is None:
            if self.obs_mask is not None:
                sampled_batch["normed_states"] = sampled_batch["normed_states"][
                    :, self.obs_mask
                ]
            masked_obs["state"] = sampled_batch["normed_states"]
        act_mask, obs_mask = None, None
        if self.fix_obs_steps:
            act_mask, obs_mask = self.act_mask, self.obs_mask
        if act_mask is None or obs_mask is None:
            if self.obs_as_global_cond:
                act_mask, obs_mask, _ = self.mask_generator(
                    traj_data.shape, self.device
                )
                self.act_mask, self.obs_mask = act_mask, obs_mask
                for key in masked_obs:
                    masked_obs[key] = masked_obs[key][:, obs_mask, ...]
            else:
                raise NotImplementedError(
                    "Not support diffuse over obs! Please set obs_as_global_cond=True"
                )

        if self.obs_encoder is not None:
            obs_fea = self.obs_encoder(masked_obs)
        else:
            obs_fea = masked_obs["state"].reshape(batch_size, -1)

        kpam_keyframe_step = (
            torch.where(
                (sampled_batch["kpam_states"]["peg_head_wrt_goal_x_dist"] < 0.011)
                & (sampled_batch["kpam_states"]["peg_head_wrt_goal_yz_dist"] < 0.01)
                & (sampled_batch["kpam_states"]["peg_wrt_goal_yz_dist"] < 0.01),
                1.0,
                0.0,
            )
            .squeeze(-1)
            .argmax(dim=-1)
        )
        kpam_act_mask = act_mask.clone()
        kpam_act_mask[
            torch.arange(batch_size, device=act_mask.device), kpam_keyframe_step
        ] = True

        loss, ret_dict = self.loss(
            x=traj_data,
            masks=sampled_batch["is_valid"],
            cond_mask=kpam_act_mask,
            global_cond=obs_fea,
        )  # TODO: local_cond, returns
        loss.backward()
        self.actor_optim.step()

        # Not implement yet
        # if self.step % self.update_ema_every == 0:
        #     self.step_ema()

        ret_dict["grad_norm_diff_model"] = np.mean(
            [
                torch.linalg.norm(parameter.grad.data).item()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
        )
        if self.obs_encoder is not None:
            ret_dict["grad_norm_diff_obs_encoder"] = np.mean(
                [
                    torch.linalg.norm(parameter.grad.data).item()
                    for parameter in self.obs_encoder.parameters()
                    if parameter.grad is not None
                ]
            )

        if self.lr_scheduler is not None:
            ret_dict["lr"] = get_mean_lr(self.actor_optim)
        ret_dict = dict(ret_dict)
        ret_dict = {"diffusion/" + key: val for key, val in ret_dict.items()}

        self.step += 1

        return ret_dict
