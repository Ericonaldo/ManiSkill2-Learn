python data_process/data_process.py --traj-path ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.h5 \
--save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5 --use-env-states --allow-failure

python data_process/data_process.py --traj-path ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.h5 \
--save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5 --use-env-states --allow-failure

python data_process/data_process.py --traj-path ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.h5 \
--save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5 --use-env-states --allow-failure

python data_process/data_process.py --traj-path ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.h5 \
--save-traj --obs-mode none --target-control-mode pd_ee_delta_pose --num-proc 5 --use-env-states --allow-failure




python tools/key_frame_identification.py --env-id=PickCube
python tools/key_frame_identification.py --env-id=LiftCube
python tools/key_frame_identification.py --env-id=StackCube
python tools/key_frame_identification.py --env-id=PegInsertionSide
python tools/key_frame_identification.py --env-id=PlugCharger



python tools/convert_state.py --env-name LiftCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_angle
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajmslearn.keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name LiftCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/LiftCube-v0/trajmslearn.keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force


python tools/convert_state.py --env-name PickCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_angle
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PickCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.angle.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_angle --using_target
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PickCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PickCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_target
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PickCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.euler.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_euler --using_target
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.euler.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.keyframes.euler.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--use-euler --extra-dim 6 --pose-dim 6
#### relative keyframe pose
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.relative_keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--keyframe-relative-pose --extra-dim 6 --pose-dim 6
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PickCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.targetpos.relative_keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--keyframe-relative-pose --extra-dim 6 --pose-dim 7


python tools/convert_state.py --env-name StackCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_angle
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/StackCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name StackCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/StackCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name StackCube-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.euler.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_euler
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.euler.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/StackCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.keyframes.euler.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--pose-dim 6
#### relative keyframe pose
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/StackCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.relative_keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--keyframe-relative-pose --extra-dim 6 --pose-dim 6
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/StackCube-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.relative_keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--keyframe-relative-pose --extra-dim 6 --pose-dim 7

python tools/convert_state.py --env-name PegInsertionSide-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_angle
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PegInsertionSide-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PegInsertionSide-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.euler.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_euler
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.euler.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.keyframes.euler.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--pose-dim 6


python tools/convert_state.py --env-name PlugCharger-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_angle
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.angle.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.keyframes.angle.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PlugCharger-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.keyframes.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force
python tools/convert_state.py --env-name PlugCharger-v0 --num-procs 10 \
--traj-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.h5 \
--json-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajectory.none.pd_ee_delta_pose.json \
--output-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.euler.rgbd.pd_ee_delta_pose.h5 \
--control-mode pd_ee_delta_pose --max-num-traj -1 --obs-mode rgbd --concat-rgbd --force --using_euler
python tools/extract_state_from_keyframe.py --num-procs 1 \
--traj-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.euler.rgbd.pd_ee_delta_pose.h5 \
--keyframe-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/keyframes.pd_ee_delta_pose.h5 \
--output-name ../ManiSkill2/demos/rigid_body/PlugCharger-v0/trajmslearn.keyframes.euler.rgbd.pd_ee_delta_pose.h5 \
--max-num-traj -1 --force \
--pose-dim 6



rsync -av --progress -r --partial --append-verify ../ManiSkill2/demos/rigid_body/LiftCube-v0/* workspace:/mnt/bn/robotics-data-hl/ManiSkill2/demos/rigid_body/LiftCube-v0
rsync -av --progress -r --partial --append-verify ../ManiSkill2/demos/rigid_body/PickCube-v0/trajmslearn.* workspace:/mnt/bn/robotics-data-hl/ManiSkill2/demos/rigid_body/PickCube-v0
rsync -av --progress -r --partial --append-verify ../ManiSkill2/demos/rigid_body/StackCube-v0/trajmslearn.* workspace:/mnt/bn/robotics-data-hl/ManiSkill2/demos/rigid_body/StackCube-v0
rsync -av --progress -r --partial --append-verify ../ManiSkill2/demos/rigid_body/PegInsertionSide-v0/trajmslearn.* workspace:/mnt/bn/robotics-data-hl/ManiSkill2/demos/rigid_body/PegInsertionSide-v0
rsync -av --progress -r --partial --append-verify ../ManiSkill2/demos/rigid_body/PlugCharger-v0/* workspace:/mnt/bn/robotics-data-hl/ManiSkill2/demos/rigid_body/PlugCharger-v0
