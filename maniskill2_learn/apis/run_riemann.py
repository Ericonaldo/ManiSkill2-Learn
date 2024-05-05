import argparse
import os
import os.path as osp
import shutil
import time
import warnings
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np

np.set_printoptions(3)
warnings.simplefilter(action="ignore")


from maniskill2_learn.utils.data import is_not_null, is_null, num_to_str
from maniskill2_learn.utils.meta import (
    Config,
    DictAction,
    add_env_var,
    collect_env,
    colored_print,
    get_logger,
    get_world_rank,
    is_debug_mode,
    log_meta_info,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified API for Training and Evaluation"
    )
    # Configurations
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "--cfg-options",
        "--opt",
        nargs="+",
        action=DictAction,
        help="Override some settings in the configuration file. The key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overridden is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    parser.add_argument("--debug", action="store_true", default=False)

    # Parameters for log dir
    parser.add_argument(
        "--work-dir", help="The directory to save logs and models", default=None
    )
    parser.add_argument("--env-id", help="Env name", default="None")
    parser.add_argument(
        "--dev",
        action="store_true",
        default=True,
        help="Add timestamp to the name of work-dir",
    )
    parser.add_argument(
        "--with-agent-type",
        default=True,
        action="store_true",
        help="Add agent type to work-dir",
    )
    parser.add_argument(
        "--agent-type-first",
        default=False,
        action="store_true",
        help="When work-dir is None, we will use agent_type/config_name or config_name/agent_type as work-dir",
    )
    parser.add_argument("--clean-up", help="Clean up the work-dir", action="store_true")

    parser.add_argument(
        "--build-replay", help="Build replay for evaluation", action="store_true"
    )

    # Specify GPU
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--num-gpus", default=None, type=int, help="Number of gpus to use"
    )
    group_gpus.add_argument(
        "-g", "--gpu-ids", default=None, type=int, nargs="+", help="ids of gpus to use"
    )
    parser.add_argument(
        "--sim-gpu-ids",
        default=None,
        type=int,
        nargs="+",
        help="ids of gpus to do simulation on; if not specified, this equals --gpu-ids",
    )

    # Resume checkpoint model
    parser.add_argument(
        "--resume-from",
        default=None,
        nargs="+",
        help="A specific checkpoint file to resume from",
    )
    parser.add_argument(
        "--resume-keys-map",
        default=None,
        nargs="+",
        action=DictAction,
        help="Specify how to change the model keys in checkpoints",
    )

    # Torch and reproducibility settings
    parser.add_argument(
        "--seed", type=int, default=None, help="Set torch and numpy random seed"
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Whether to use benchmark mode in cudnn.",
    )

    args = parser.parse_args()

    # Merge cfg with args.cfg_options
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        for key, value in args.cfg_options.items():
            try:
                value = eval(value)
                args.cfg_options[key] = value
            except Exception:
                pass
        cfg.merge_from_dict(args.cfg_options)

    args.with_agent_type = args.with_agent_type or args.agent_type_first
    for key in ["work_dir", "env_cfg", "extra_env_cfg", "rollout_cfg", "resume_from"]:
        cfg[key] = cfg.get(key, None)
    if cfg.extra_env_cfg is not None and cfg.env_cfg is not None:
        tmp = cfg.extra_env_cfg
        cfg.extra_env_cfg = deepcopy(cfg.env_cfg)
        cfg.extra_env_cfg.update(tmp)
    if args.debug:
        os.environ["PYRL_DEBUG"] = "True"
    elif "PYRL_DEBUG" not in os.environ:
        os.environ["PYRL_DEBUG"] = "False"
    if args.seed is None:
        args.seed = np.random.randint(2**32 - int(1e8))
    args.mode = "eval"
    return args, cfg


def build_work_dir():
    if is_null(args.work_dir):
        root_dir = "./logs"
        env_name = (
            cfg.env_cfg.get("env_name", None)
            if is_not_null(cfg.env_cfg)
            else args.env_id
        )
        config_name = osp.splitext(osp.basename(args.config))[0]
        folder_name = env_name if is_not_null(env_name) else config_name
        if args.with_agent_type:
            if args.agent_type_first:
                args.work_dir = osp.join(root_dir, agent_type, folder_name)
            else:
                args.work_dir = osp.join(root_dir, folder_name, agent_type)
        else:
            args.work_dir = osp.join(root_dir, folder_name)
    elif args.with_agent_type:
        if args.agent_type_first:
            colored_print(
                "When you specify the work dir path, the agent type cannot be at the beginning of the path!",
                level="warning",
            )
        args.work_dir = osp.join(args.work_dir, agent_type)

    args.work_dir = osp.join(args.work_dir, cfg.get("workdir", "default"))

    if args.dev:
        args.work_dir = osp.join(args.work_dir, args.timestamp)

    os.makedirs(osp.abspath(args.work_dir), exist_ok=True)


def get_python_env_info():
    env_info_dict = collect_env()
    num_gpus = env_info_dict["Num of GPUs"]

    if is_not_null(args.num_gpus):
        args.gpu_ids = list(range(args.num_gpus))
    else:
        args.gpu_ids = []
    if len(args.gpu_ids) == 0 and num_gpus > 0:
        colored_print(
            f"We will use cpu, although we have {num_gpus} gpus available!",
            level="warning",
        )

    if len(args.gpu_ids) > 1:
        colored_print(
            "Multiple GPU is not supported; we will use the first GPU!",
            level="warning",
        )
        args.gpu_ids = args.gpu_ids[:1]
    args.env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])


def init_torch(args):
    import torch

    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    rank = get_world_rank()
    if args.gpu_ids is not None and len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[rank])
        torch.set_num_threads(1)

    if is_debug_mode():
        torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_collect(evaluator, args, cfg):
    import torch.distributed as dist

    from maniskill2_learn.env import save_eval_statistics
    from maniskill2_learn.methods.builder import build_agent
    from maniskill2_learn.utils.torch import BaseAgent, load_checkpoint

    logger = get_logger()
    logger.info("Initialize torch!")
    init_torch(args)
    logger.info("Finish Initialize torch!")

    logger.info("Build diffusion agent!")
    diff_agent = build_agent(cfg.diff_agent_cfg)
    assert (
        diff_agent is not None
    ), f"Diffusion agent type {cfg.agent_cfg.type} is not valid!"

    device = "cpu" if len(args.gpu_ids) == 0 else "cuda"
    diff_agent = diff_agent.float().to(device)
    assert isinstance(
        diff_agent, BaseAgent
    ), "The agent object should be an instance of BaseAgent!"

    assert is_not_null(cfg.resume_from), cfg.resume_from
    logger.info("Resume diffusion agent with checkpoint!")
    for ckpt_file in cfg.resume_from:
        load_checkpoint(
            diff_agent, ckpt_file, device, keys_map=args.resume_keys_map, logger=logger
        )

    logger.info("Build riemann expert!")
    cfg.agent_cfg["diff_model"] = diff_agent
    agent = build_agent(cfg.agent_cfg)
    assert agent is not None, f"Agent type {cfg.agent_cfg.type} is not valid!"
    agent = agent.float().to(device)
    logger.info(agent)
    logger.info(
        f'Num of parameters: {num_to_str(agent.num_trainable_parameters, unit="M")}, Model Size: {num_to_str(agent.size_trainable_parameters, unit="M")}'
    )
    assert isinstance(
        agent, BaseAgent
    ), "The agent object should be an instance of BaseAgent!"

    logger.info(f"Work directory of this run {args.work_dir}")
    if len(args.gpu_ids) > 0:
        logger.info(f"Train over GPU {args.gpu_ids}!")
    else:
        logger.info("Train over CPU!")

    agent.eval()
    agent.set_mode("test")
    info = evaluator.run(agent, work_dir=work_dir, **cfg.eval_cfg)
    save_eval_statistics(work_dir, logger, **info)
    agent.train()
    agent.set_mode("train")

    if len(args.gpu_ids) > 1:
        dist.destroy_process_group()


def main(args, cfg):
    if len(args.gpu_ids) > 1:
        raise NotImplementedError(f"GPU IDs: {args.gpu_ids}! Not supported yet!")

    import numpy as np

    set_random_seed(args.seed)

    if is_not_null(args.resume_from):
        if is_not_null(cfg.resume_from):
            colored_print(
                f"The resumed checkpoint from the config file is overwritten by {args.resume_from}!",
                level="warning",
            )
        cfg.resume_from = args.resume_from

    if is_not_null(cfg.env_cfg) and len(args.gpu_ids) > 0:
        if args.sim_gpu_ids is not None:
            assert len(args.sim_gpu_ids) == len(
                args.gpu_ids
            ), "Number of simulation gpus should be the same as the number of training gpus!"
        else:
            args.sim_gpu_ids = args.gpu_ids
        cfg.env_cfg.device = f"cuda:{args.gpu_ids[0]}"

    work_dir = args.work_dir
    logger_file = osp.join(work_dir, f"{args.timestamp}-{args.name_suffix}.log")
    logger = get_logger(
        name=None, log_file=logger_file, log_level=cfg.get("log_level", "INFO")
    )

    if is_debug_mode():
        dash_line = "-" * 60 + "\n"
        logger.info(
            "Environment info:\n" + dash_line + args.env_info + "\n" + dash_line
        )

    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Set random seed to {args.seed}")

    # Build evaluation module
    from maniskill2_learn.env import build_evaluation

    logger.info("Build evaluation!")
    eval_cfg = cfg.eval_cfg
    # Evaluation environment setup can be different from the training set-up. (Like early-stop or object sets)
    if eval_cfg.get("env_cfg", None) is None:
        eval_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
    else:
        tmp = eval_cfg["env_cfg"]
        eval_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
        eval_cfg["env_cfg"].update(tmp)
    if eval_cfg.get("extra_env_cfg", None) is None:
        eval_cfg["extra_env_cfg"] = deepcopy(cfg.extra_env_cfg)
    else:
        tmp = eval_cfg["extra_env_cfg"]
        eval_cfg["extra_env_cfg"] = deepcopy(cfg.extra_env_cfg)
        eval_cfg["extra_env_cfg"].update(tmp)

    get_logger().info(f"Building evaluation: eval_cfg: {eval_cfg}")
    eval_cfg["seed"] = args.seed
    if args.seed is None:
        eval_cfg["seed"] = np.random.randint(0, int(1e9))
    evaluator = build_evaluation(eval_cfg)

    # Get environments information for agents
    from maniskill2_learn.env import get_env_info

    if hasattr(evaluator, "vec_env"):
        env_params = get_env_info(cfg.env_cfg, evaluator.vec_env)
    else:
        env_params = get_env_info(cfg.env_cfg)

    obs_shape = env_params["obs_shape"]
    action_shape = env_params["action_shape"]
    logger.info(
        f'Got shapes from env_params: state shape:{env_params["obs_shape"]}, action shape:{env_params["action_shape"]}'
    )

    cfg.agent_cfg["env_params"] = env_params
    cfg.diff_agent_cfg["env_params"] = env_params
    if is_not_null(obs_shape) or is_not_null(action_shape):
        from maniskill2_learn.networks.utils import (
            get_kwargs_from_shape,
            replace_placeholder_with_args,
        )

        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)
    logger.info(f"Final agent config:\n{cfg.agent_cfg}")

    # Output version of important packages
    log_meta_info(logger)
    main_collect(evaluator, args, cfg)
    evaluator.close()
    logger.info("Close evaluator object")


if __name__ == "__main__":
    # Remove mujoco_py lock
    mjpy_lock = (
        Path(gym.__file__).parent.parent / "mujoco_py/generated/mujocopy-buildlock.lock"
    )
    if mjpy_lock.exists():
        os.remove(str(mjpy_lock))

    add_env_var()

    args, cfg = parse_args()
    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    agent_type = cfg.agent_cfg.type

    build_work_dir()
    get_python_env_info()

    work_dir = args.work_dir
    # Always clean up for evaluation
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir

    logger_name = (
        cfg.env_cfg.env_name if is_not_null(cfg.env_cfg) else cfg.agent_cfg.type
    )
    args.name_suffix = f"{args.mode}"
    os.environ["PYRL_LOGGER_NAME"] = f"{logger_name}-{args.name_suffix}"
    cfg.dump(osp.join(work_dir, f"{args.timestamp}-{args.name_suffix}.py"))

    main(args, cfg)
