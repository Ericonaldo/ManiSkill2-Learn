# ManiSkill2 Dataset

## Installation
1. Follow the instruction on [ManiSkill2 github](https://github.com/haosulab/ManiSkill2) and install ManiSkill2. You may install from source as below.

    ```
    git clone https://github.com/haosulab/ManiSkill2.git
    cd ManiSkill2 && pip install -e .
    ```
2. Download the asset.
    ```
    python -m mani_skill2.utils.donwload_asset all
    ```
3. Download the demos. 
    ```
    python -m mani_skill2.utils.download_demo {env_id} -o demos 
    ```
    In this benchmark, we foucs on the following environments: *AssemblingKits-v0, PegInsertionSide-v0, PlugCharger-v0, TurnFaucet-v0 x 2, Hang-v0, Pour-v0, LiftCube-v0*. For all environments, see [download_demo.py](https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/utils/download_demo.py).

## Processing
The downloaded demos only contains environment information and do not have observations or actions for training. We need to replay the demos to extract them.
1. Export the asset directory.
    ```
    export MS2_ASSET_DIR=/path/to/ManiSkill2/data/
    ```
2. Extract the episodes. Here we show the examples of extracting from the episodes of LiftCube-v0.
    ```
    python data_process.py --traj-path $MS2_DEMO_DIR/rigid_body/LiftCube-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
    ```
    Note that the newly generated trajectory h5 and json file will be saved in the same directory of the trajectory path. So put the demos in a directory with large space.

    ``obs-mode`` can have various options: ``image``, ``pointcloud``, ``rgbd``, ``state``, ``state_dict``. For more details, see the [ManiSkill website](https://haosulab.github.io/ManiSkill2/concepts/observation.html).

## DataLoading
The main code for loading the data is ``dataset.py``.

## Action space
*Note: ManiSkill2 uses a PD controller to drive motors to achieve target joint positions. In the case of controlling ee delta pose, the controller may not be able to achieve exactly at the target joint position.*

Action space is 7-dim.
``act[0:3]`` refers to ``xyz`` ; ``act[3:6]`` refers to axis-angle; ``act[6]`` refers to the gripper action
**All dimensions are clipped to (-1, 1).**
The range of first 6 dimensions are (-0.1, 0.1).
The range of the last dimension is (-0.04, 0.04).