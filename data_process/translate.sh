# Rigid body
## image 
# python data_process.py --traj-path ../demos/rigid_body/LiftCube-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/AssemblingKits-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10 
# python data_process.py --traj-path ../demos/rigid_body/PegInsertionSide-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PickCube-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PlugCharger-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PickClutterYCB-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10 & 
# python data_process.py --traj-path ../demos/rigid_body/PickSingleYCB-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10 
# rsync -av --progress -r /media/wuhongtao123/datadisk/data/maniskill2/demos/rigid_body/PickClutterYCB-v0 /media/wuhongtao123/datadisk/data/maniskill2/demos/rigid_body/PickSingleYCB-v0 il:/mnt/bn/robotics-data-hl/maniskill2_data/rigid_body/
# rm /media/wuhongtao123/datadisk/data/maniskill2/demos/rigid_body/PickClutterYCB-v0/*image* & rm /media/wuhongtao123/datadisk/data/maniskill2/demos/rigid_body/PickSingleYCB-v0/*image*

# pointcloud
# python data_process.py --traj-path ../demos/rigid_body/LiftCube-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
# mv ../demos/rigid_body/LiftCube-v0/trajectory.pointcloud* /home/eric/bytenas/ManiSkill2-Learn/demos/rigid_body/LiftCube-v0/; rm ../demos/rigid_body/LiftCube-v0/trajectory.pointcloud* &
# python data_process.py --traj-path ../demos/rigid_body/AssemblingKits-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
# mv ../demos/rigid_body/AssemblingKits-v0/trajectory.pointcloud* /home/eric/bytenas/ManiSkill2-Learn/demos/rigid_body/AssemblingKits-v0/; rm ../demos/rigid_body/AssemblingKits-v0/trajectory.pointcloud*&
# python data_process.py --traj-path ../demos/rigid_body/PandaAvoidObstacles-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PegInsertionSide-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PickClutterYCB-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PickCube-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
# mv ../demos/rigid_body/PickCube-v0/trajectory.pointcloud* /home/eric/bytenas/ManiSkill2-Learn/demos/rigid_body/PickCube-v0/; rm ../demos/rigid_body/PickCube-v0/trajectory.pointcloud*&
# python data_process.py --traj-path ../demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 5
# mv ../demos/rigid_body/PickSingleEGAD-v0/trajectory.pointcloud* /home/eric/bytenas/ManiSkill2-Learn/demos/rigid_body/PickSingleEGAD-v0/; rm ../demos/rigid_body/PickSingleEGAD-v0/trajectory.pointcloud*&
python data_process.py --traj-path ../demos/rigid_body/PickSingleYCB-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PlugCharger-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/rigid_body/TurnFaucet-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10


## TODO
# rgbd
# python data_process.py --traj-path ../demos/rigid_body/LiftCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/AssemblingKits-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PandaAvoidObstacles-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PegInsertionSide-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PickClutterYCB-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PickCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/rigid_body/PickSingleYCB-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PlugCharger-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/rigid_body/TurnFaucet-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10


# none
# python data_process.py --traj-path ../demos/rigid_body/LiftCube-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/AssemblingKits-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PandaAvoidObstacles-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PegInsertionSide-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PickClutterYCB-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PickCube-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
python data_process.py --traj-path ../demos/rigid_body/PickSingleYCB-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PlugCharger-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
python data_process.py --traj-path ../demos/rigid_body/TurnFaucet-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5
# python data_process.py --traj-path ../demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 10

python tools/convert_state.py --env-name PickCube-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/PickCube-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd


python tools/convert_state.py --env-name StackCube-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/StackCube-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

python tools/convert_state.py --env-name LiftCube-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/LiftCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/LiftCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/LiftCube-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

python tools/convert_state.py --env-name AssemblingKits-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/AssemblingKits-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/AssemblingKits-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/AssemblingKits-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

python tools/convert_state.py --env-name PandaAvoidObstacles-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/PandaAvoidObstacles-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/PandaAvoidObstacles-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/PandaAvoidObstacles-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd


python tools/convert_state.py --env-name PegInsertionSide-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/PegInsertionSide-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

python tools/convert_state.py --env-name PickClutterYCB-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/PickClutterYCB-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/PickClutterYCB-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/PickClutterYCB-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

python tools/convert_state.py --env-name PlugCharger-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/PlugCharger-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

python tools/convert_state.py --env-name StacPickSingleEGADkCube-v0 --num-procs 5 \
--traj-name ./demos/rigid_body/PickSingleEGAD-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ./demos/rigid_body/PickSingleEGAD-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ./demos/rigid_body/PickSingleEGAD-v0/trajmslearn.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -100000 --obs-mode rgbd

## Soft body
# image
# python data_process.py --traj-path ../demos/soft_body/Excavate-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/soft_body/Fill-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/soft_body/Hang-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/soft_body/Pinch-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/soft_body/Pour-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# python data_process.py --traj-path ../demos/soft_body/Write-v0/trajectory.h5 --save-traj --obs-mode image --target-control-mode pd_ee_delta_pose --num-proc 10
# pointcloud
python data_process.py --traj-path ../demos/soft_body/Fill-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Hang-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Pinch-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Pour-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Write-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Excavate-v0/trajectory.h5 --save-traj --obs-mode pointcloud --target-control-mode pd_ee_delta_pose --num-proc 10
# rgbd
python data_process.py --traj-path ../demos/soft_body/Fill-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Hang-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Pinch-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Pour-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Write-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10
python data_process.py --traj-path ../demos/soft_body/Excavate-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-proc 10