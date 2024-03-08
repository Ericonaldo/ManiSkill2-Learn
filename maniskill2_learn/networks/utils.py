from maniskill2_learn.utils.meta import ConfigDict, Config
from numbers import Number
from copy import deepcopy
import numpy as np
from torch.nn.parameter import Parameter
import torch


def combine_obs_with_action(obs, action=None):
    if action is None:
        return obs
    elif isinstance(obs, dict):
        obs = obs.copy()
        if "state" not in obs:
            # For DM Control
            obs["state"] = action
        else:
            # For ManiSkill
            obs["state"] = torch.cat([obs["state"], action], dim=-1)
        return obs
    else:
        return torch.cat([obs, action], dim=-1)


def get_kwargs_from_shape(obs_shape, action_shape):
    PCD_KEYS = [
        "pointcloud",
        "full_pcd",
        "no_robot",
        "handle_only",
        "fused_pcd",
        "fused_ball_pcd",
        "pointcloud_3d_ann",
        "particles",
    ]
    IMAGE_KEYS = [
        "rgb",
        "rgbd",
        "depth",
        "base_camera_rgbd",
        "hand_camera_rgbd",
        "base_camera_rgbd",
        "hand_camera_rgbd",
    ]  # Add your own keys if you have different obs space
    replaceable_kwargs = {}
    if action_shape is not None:
        replaceable_kwargs["action_shape"] = deepcopy(action_shape)

    for key in obs_shape.keys():
        if isinstance(obs_shape[key], list) and (
            not isinstance(obs_shape[key][0], int)
        ):
            obs_shape[key] = obs_shape[key][0]

    if isinstance(obs_shape, dict):
        if "state" in obs_shape:
            replaceable_kwargs["agent_shape"] = obs_shape["state"]
            # replaceable_kwargs["agent_shape"] = obs_shape["state"] - 9 # We remove the dimension of velocity
        elif "agent" in obs_shape:
            replaceable_kwargs["agent_shape"] = obs_shape["agent"]
            # replaceable_kwargs["agent_shape"] = obs_shape["agent"] - 9 # We remove the dimension of velocity

        if "hand_pose" in obs_shape:
            replaceable_kwargs["nhand"] = obs_shape["hand_pose"][1]

        if "xyz" in obs_shape:
            visual_key = "pointcloud"
            visual_shape = obs_shape
        else:
            try:
                visual_key = [
                    name
                    for name in obs_shape.keys()
                    if name in PCD_KEYS or name in IMAGE_KEYS
                ][0]
            except IndexError:
                visual_key = None
            visual_shape = obs_shape

        if visual_key is not None:
            if visual_key in PCD_KEYS:
                # For mani_skill point cloud input
                pcd_all_channel, pcd_xyz_rgb_channel = 0, 0
                for name in ["xyz", "rgb"]:
                    if name in visual_shape:
                        pcd_xyz_rgb_channel += visual_shape[name][-1]
                        pcd_all_channel += visual_shape[name][-1]
                if "seg" in visual_shape:
                    replaceable_kwargs["num_objs"] = visual_shape["seg"][-1]
                    pcd_all_channel += visual_shape["seg"][-1]
                if "target_object_point" in visual_shape:
                    pcd_all_channel += visual_shape["target_object_point"]
                replaceable_kwargs["pcd_all_channel"] = pcd_all_channel
                replaceable_kwargs["pcd_xyz_rgb_channel"] = pcd_xyz_rgb_channel
                replaceable_kwargs["pcd_xyz_channel"] = 3
                for name in obs_shape:
                    replaceable_kwargs["pcd_{}_shape".format(name)] = obs_shape[name]

            elif visual_key in IMAGE_KEYS:
                # For new maniskill callback envs
                if len(visual_shape[visual_key]) == 3:  # NOTE: Be careful!
                    num_images = 1
                else:
                    # assert len(visual_shape[visual_key]) == 4, f"You need to provide either 3-dim or 5-dim inputs! The input shape is {visual_shape[visual_key]}!"
                    num_images = visual_shape[visual_key][0]  # [K, C, N, M]
                replaceable_kwargs["image_size"], replaceable_kwargs["num_images"] = (
                    visual_shape[visual_key][-2:],
                    num_images,
                )
                # if isinstance(replaceable_kwargs["image_size"], list) and (not isinstance(replaceable_kwargs["image_size"][0], int)):
                #     replaceable_kwargs["image_size"] = replaceable_kwargs["image_size"][0]
                replaceable_kwargs["num_pixels"] = np.prod(replaceable_kwargs["image_size"])
                replaceable_kwargs["image_channels"] = (
                    sum(
                        [
                            visual_shape[name][-3]
                            for name in IMAGE_KEYS
                            if name in visual_shape
                        ]
                    )
                    * num_images  # NOTE: Be careful!
                )
                if "depth" in visual_shape and "seg" in visual_shape:
                    replaceable_kwargs["seg_per_image"] = visual_shape["seg"][-3]
    else:
        replaceable_kwargs["obs_shape"] = deepcopy(obs_shape)

    replaceable_kwargs["agent_shape+action_shape"] = (
        replaceable_kwargs["agent_shape"] + replaceable_kwargs["action_shape"]
    )

    print("Got replaceable kwargs, ", replaceable_kwargs)

    return replaceable_kwargs


def replace_placeholder_with_args(parameters, **kwargs):
    if parameters is None:
        return None
    elif isinstance(parameters, ConfigDict):
        for key, v in parameters.items():
            parameters[key] = replace_placeholder_with_args(v, **kwargs)
        return parameters
    elif isinstance(parameters, Config):
        for key, v in parameters.dict().items():
            parameters[key] = replace_placeholder_with_args(v, **kwargs)
        return parameters
    elif isinstance(parameters, (tuple, list)):
        type_of_parameters = type(parameters)
        parameters = list(parameters)
        for i, parameter in enumerate(parameters):
            parameters[i] = replace_placeholder_with_args(parameter, **kwargs)
        return type_of_parameters(parameters)
    elif isinstance(parameters, Number):
        return parameters
    elif isinstance(parameters, str):
        for key in kwargs:
            if key in parameters:
                parameters = parameters.replace(key, str(kwargs[key]))
        try:
            ret = eval(parameters)
            if callable(ret):
                return parameters
            else:
                return ret
        except:
            return parameters
    else:
        return parameters
