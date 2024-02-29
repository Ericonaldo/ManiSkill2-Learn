import h5py
from maniskill2_learn.utils.data import DictArray, GDict

traj_name = "../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.keyframes.angle.rgbd.pd_ee_delta_pose.h5"
input_h5 = h5py.File(traj_name, "r")
trajectory = GDict.from_hdf5(input_h5["traj_0"])

import numpy as np
from PIL import Image

image_array = np.array(trajectory["obs"]["base_camera_rgbd"])
image_array = np.transpose(image_array, (0, 2, 3, 1))
print(image_array.shape)

# Iterate over the array and save each image
for i in range(image_array.shape[0]):
    print(image_array[i][:, :, :3])
    image = Image.fromarray(image_array[i][:, :, :3])
    image.save(f"image_{i}.png")
