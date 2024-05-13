from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.demo_file = "demo_4096_s100.npz"
    config.object_name = "peg"
    config.act_type = "insert"
    config.pred_type = "target"

    config.seg = ConfigDict()
    config.seg.seed = 100
    config.seg.log_dir_format = "seg-{object_name}-{act_type}-{pred_type}/{seg.seed}"
    config.seg.model = "SE3SegNet"
    config.seg.data_aug = False
    config.seg.aug_methods = []
    config.seg.random_drop = False
    config.seg.draw_pcd = False
    config.seg.mask_part = False
    config.seg.train_batch_size = 1
    config.seg.test_batch_size = 1
    config.seg.train_demo_ratio = 0.8
    config.seg.lr = 1e-3
    config.seg.epoch = 1000
    # config.seg.voxel_size = 0.01
    # config.seg.radius_threshold = 0.07
    # config.seg.resume_from = "params/target_segnet.pth"

    config.seg.voxel_size = 0.005
    config.seg.radius_threshold = 0.05
    config.seg.resume_from = "params/target_segnet_v0.005.pth"

    config.mani = ConfigDict()
    config.mani.seed = 100
    config.mani.log_dir_format = "mani-{object_name}-{act_type}-{pred_type}-d_{mani.distance_threshold}-f_{mani.feature_point_radius}/{mani.seed}"
    config.mani.model = "SE3ManiNet"
    config.mani.data_aug = False
    config.mani.aug_methods = []
    config.mani.ref_point = "gt"
    config.mani.random_drop = True
    config.mani.remain_point_ratio = [0.5, 0.9]
    config.mani.draw_pcd = False
    config.mani.mask_part = True
    config.mani.ori_weight = 0.1
    config.mani.train_batch_size = 1
    config.mani.test_batch_size = 1
    config.mani.train_demo_ratio = 0.8
    config.mani.lr = 5e-4
    config.mani.epoch = 1000
    # [0.16, 0.1]
    config.mani.distance_threshold = 0.16
    config.mani.pos_warmup_epoch = 1
    config.mani.voxel_size = 0.01
    config.mani.radius_threshold = 0.05
    config.mani.feature_point_radius = 0.05
    config.mani.resume_from = "params/target_maninet_jointtrain.pth"

    return config
