# Create a json which contains path to all the trajectories

import json
import os

tasks = [
    "rigid_body/AssemblingKits-v0",
    "rigid_body/StackCube-v0",
    "rigid_body/PegInsertionSide-v0",
    "rigid_body/PlugCharger-v0",
    "rigid_body/TurnFaucet-v0",
]
obs_mode = "image"
action_mode = "ee_delta_pose"

DATA_DIR = f"/media/wuhongtao123/ab9fc4e0-ef39-424c-ae53-28ed143bc942/data/maniskill2"
json_path = os.path.join(DATA_DIR, f"multi_tasks-{obs_mode}-{action_mode}.json")

traj_path_dict = dict()
idx = 0
for task in tasks:
    task_dir = os.path.join(DATA_DIR, "demos", task)
    files = os.listdir(task_dir)
    for file in files:
        if (obs_mode in file) and (action_mode in file) and (".h5" in file):
            traj_path_dict[idx] = os.path.join(DATA_DIR, file)
            idx += 1
with open(json_path, "w") as f:
    json.dump(traj_path_dict, f)
