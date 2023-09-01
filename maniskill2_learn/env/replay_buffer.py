import numpy as np
import os.path as osp
from typing import Union
from tqdm import tqdm
from itertools import count
from h5py import File
import _thread, threading
import torch
import copy

from maniskill2_learn.utils.meta import get_filename_suffix, get_total_memory, get_memory_list, get_logger, TqdmToLogger, parse_files
from maniskill2_learn.utils.data import is_seq_of, DictArray, GDict, is_h5, is_null, DataCoder, is_not_null
from maniskill2_learn.utils.file import load, load_items_from_record, get_index_filenames, get_total_size, FileCache, is_h5_traj, decode_items
from maniskill2_learn.utils.file.cache_utils import META_KEYS
from .builder import REPLAYS, build_sampling
from .sampling_strategy import TStepTransition
from collections import deque

@REPLAYS.register_module()
class ReplayMemory:
    """
    This replay buffer is designed for RL, BRL.
    Replay buffer uses dict-array as basic data structure, which can be easily saved as hdf5 file.
    Also it utilize a asynchronized memory cache system to speed up the file loading process.
    See dict_array.py for more details.

    Two special keys for multiprocess and engineering usage
        is_truncated: to indicate if the trajectory is truncated here and the next sample from the same worker is from another trajectory.
        woker_index: to indicate which process generate this sample. Used in recurrent policy.
        is_valid: to indicate if this sample is useful. Used in recurrent policy.
    """

    def __init__(
        self,
        capacity,
        using_depth=True,
        sampling_cfg=dict(type="OneStepTransition"),
        keys=None,
        keys_map=None,
        data_coder_cfg=None,
        buffer_filenames=None,
        cache_size=8192,
        num_samples=-1,
        num_procs=4,
        synchronized=True,  # For debug only which is slower than asynchronized file loading and data augmentation
        dynamic_loading=None,
        auto_buffer_resize=True,
        deterministic_loading=False,
        max_threads=10,
        sample_traj=False,
    ):
        # capacity: the size of replay buffer, -1 means we will recompute the buffer size with files for initial replay buffer.
        assert capacity > 0 or buffer_filenames is not None
        # assert sampling_cfg is not None, "Please provide a valid sampling strategy over replay buffer!"

        self.horizon = 1
        self.future_action_len = 0
        self.sample_traj = sample_traj
        self.using_depth = using_depth
        if buffer_filenames is not None and len(buffer_filenames) > 0:
            self.horizon = sampling_cfg.get("horizon", 1)
            self.future_action_len = sampling_cfg.get("future_action_len", 0)

        if buffer_filenames is not None:
            logger = get_logger()
            buffer_filenames = parse_files(buffer_filenames)
            if deterministic_loading:
                logger.warning("Sort files and change sampling strategy!")
                sampling_cfg["no_random"] = True
                buffer_filenames = sorted(buffer_filenames)
        data_coder = None if is_null(data_coder_cfg) else DataCoder(**data_coder_cfg)
        self.file_loader = None
        if buffer_filenames is not None and len(buffer_filenames) > 0:
            logger.info(f"Load {len(buffer_filenames)} files!")
            data_size = get_total_size(buffer_filenames, num_samples=num_samples)
            self.data_size = data_size
            logger.info(f"Load {len(buffer_filenames)} files with {data_size} samples in total!")

            # For supervised learning with variable number of points.
            without_cache = data_coder is not None and data_coder.var_len_item
            # Cache utils does not support var input length recently
            if capacity < 0:
                capacity = data_size
                logger.info(f"Recomputed replay buffer size is {capacity}!")
            self.dynamic_loading = dynamic_loading if dynamic_loading is not None else (capacity < data_size)
            if self.dynamic_loading and cache_size != capacity:
                logger.warning("You should use same the cache_size as the capacity when dynamically loading files!")
                cache_size = capacity
            if not self.dynamic_loading and keys is not None:
                logger.warning("Some important keys may be dropped in buffer and the buffer cannot be extended!")
            if not without_cache:
                self.file_loader = FileCache(
                    buffer_filenames,
                    min(cache_size, capacity),
                    keys,
                    data_coder,
                    num_procs,
                    synchronized=synchronized,
                    num_samples=num_samples,
                    horizon=self.horizon,
                    keys_map=keys_map,
                    deterministic_loading=deterministic_loading,
                )
                logger.info("Finish building file cache!")
            else:
                logger.info("Load without cache!")
        else:
            self.dynamic_loading = False
        if sampling_cfg is not None:
            sampling_cfg["capacity"] = capacity
            self.sampling = build_sampling(sampling_cfg)
        else:
            self.sampling = None

        if self.dynamic_loading:
            self.sampling.with_replacement = False

        self.traj_sampling = None
        if self.sample_traj:
            sampling_cfg["horizon"] = -1
            self.traj_sampling = build_sampling(sampling_cfg)
            self.prefetched_traj_queue = deque(maxlen=max_threads)

        self.capacity = capacity
        self.auto_buffer_resize = auto_buffer_resize
        self.memory = None
        self.position = 0
        self.running_count = 0
        self.reset()

        self.prefetched_data_queue = deque(maxlen=max_threads)
        self.max_threads = max_threads
        self.thread_count = 0

        if buffer_filenames is not None and len(buffer_filenames) > 0:
            if self.dynamic_loading:
                self.file_loader.run(auto_restart=False)
                items = self.file_loader.get()
                self.push_batch(items)
                self.file_loader.run(auto_restart=False)
            else:
                logger.info("Load all the data at one time!")
                if not without_cache:
                    tqdm_obj = tqdm(file=TqdmToLogger(), mininterval=10, total=self.data_size)
                    while True:
                        self.file_loader.run(auto_restart=False)
                        items = self.file_loader.get()

                        if items is None:
                            break
                        self.push_batch(items)
                        tqdm_obj.update(len(items))
                else:
                    logger.info(f"Loading full dataset without cache system!")
                    for filename in tqdm(file=TqdmToLogger(), mininterval=60)(buffer_filenames):
                        file = File(filename, "r")
                        traj_keys = [key for key in list(file.keys()) if key not in META_KEYS]
                        traj_keys = sorted(traj_keys)
                        if num_samples > 0:
                            traj_keys = traj_keys[:num_samples]
                        data = DictArray.from_hdf5(filename, traj_keys)
                        if keys is not None:
                            data = data.select_by_keys(keys)
                        if is_not_null(data_coder):
                            data = data_coder.decode(data)
                        data = data.to_two_dims()
                        self.push_batch(data)
                logger.info(f"Finish file loading! Buffer length: {len(self)}, buffer size {self.memory.nbytes_all / 1024 / 1024} MB!")
                logger.info(f"Len of sampling buffer: {len(self.sampling)}")

    def prefetched_data(self, whole_traj=False):
        data_queue = self.prefetched_data_queue
        if self.sample_traj and whole_traj:
            data_queue = self.prefetched_traj_queue
        if len(data_queue):
            return data_queue.popleft()
        return None

    def __getitem__(self, key):
        return self.memory[key]

    def __setitem__(self, key, value):
        self.memory[key] = value

    def __getattr__(self, key):
        return getattr(self.memory, key, None)

    def __len__(self):
        return min(self.running_count, self.capacity)

    def reset(self):
        self.position = 0
        self.running_count = 0
        # self.memory = None
        if self.sampling is not None:
            self.sampling.reset()

    def push(self, item):
        if not isinstance(item, DictArray):
            item = DictArray(item, capacity=1)
        self.push_batch(item)

    def push_batch(self, items: Union[DictArray, dict]):
        if not isinstance(items, DictArray):
            items = DictArray(items)
        if len(items) > self.capacity:
            items = items.slice(slice(0, self.capacity))

        if "worker_indices" not in items:
            items["worker_indices"] = np.zeros([len(items), 1], dtype=np.int32)
        if "is_truncated" not in items:
            items["is_truncated"] = np.zeros([len(items), 1], dtype=np.bool_)

        if self.memory is None:
            # Init the whole buffer
            self.memory = DictArray(items.slice(0), capacity=self.capacity)
        if self.position + len(items) > self.capacity:
            # Deal with buffer overflow
            final_size = self.capacity - self.position
            self.push_batch(items.slice(slice(0, final_size)))
            self.position = 0
            self.push_batch(items.slice(slice(final_size, len(items))))
        else:
            self.memory.assign(slice(self.position, self.position + len(items)), items)
            self.running_count += len(items)
            self.position = (self.position + len(items)) % self.capacity
            if self.sampling is not None:
                self.sampling.push_batch(items)
            if self.traj_sampling is not None:
                self.traj_sampling.push_batch(items)

    def update_all_items(self, items):
        self.memory.assign(slice(0, len(items)), items)

    def tail_mean(self, num):
        return self.memory.slice(slice(len(self) - num, len(self))).to_gdict().mean()

    def get_all(self, key=None, sub_key=None):
        # Return all elements in replay buffer
        ret = self.memory.slice(slice(0, len(self)))
        if key is not None:
            if sub_key is not None:
                # if key == "obs" and isinstance(ret["obs"][sub_key], torch.Tensor):
                #     ret["obs"][sub_key] = torch.cat([ret["obs"][sub_key][...,:9], ret["obs"][sub_key][...,18:]], axis=-1)
                # elif key == "obs" and isinstance(ret["obs"][key], np.ndarray):
                #     ret["obs"][sub_key] = np.concatenate([ret["obs"][sub_key][...,:9], ret["obs"][sub_key][...,18:]], axis=-1)
                return ret[key][sub_key]
            return ret[key]
        return ret

    def to_hdf5(self, file, with_traj_index=False):
        data = self.get_all()
        if with_traj_index:
            # Save the whole replay buffer into one trajectory.
            # TODO: Parse the trajectories in replay buffer.
            data = GDict({"traj_0": data.memory})
        data.to_hdf5(file)

    def pre_fetch(self, batch_size, auto_restart=True, drop_last=True, device=None, obs_mask=None, action_normalizer=None, obsact_normalizer=None, mode="train", whole_traj=False, keyframe_type="gpt"):
        if self.dynamic_loading and not drop_last:
            assert self.capacity % batch_size == 0

        sampler = self.sampling
        data_queue = self.prefetched_data_queue
        if self.sample_traj and whole_traj:
            sampler = self.traj_sampling
            data_queue = self.prefetched_traj_queue

        batch_idx, is_valid, ret_len = sampler.sample(batch_size, drop_last=drop_last, auto_restart=auto_restart and not self.dynamic_loading, padded_size=self.horizon)
        if batch_idx is None:
            # without replacement only
            if auto_restart or self.dynamic_loading:
                items = self.file_loader.get()
                if items is None:
                    return None
                assert self.position == 0, "cache size should equals to buffer size"
                sampler.reset()
                self.push_batch(items)
                self.file_loader.run(auto_restart=auto_restart)
                batch_idx, is_valid = sampler.sample(batch_size, drop_last=drop_last, auto_restart=auto_restart and not self.dynamic_loading)
            else:
                return None

        ret = self.memory.take(batch_idx)
        if "obs" in ret.keys():
            for key in ret["obs"].keys():
                if not self.using_depth:
                    if isinstance(ret["obs"][key], (list,tuple)):
                        ret["obs"][key] = ret["obs"][key][0]
                    if "rgbd" in key and (not self.using_depth):
                        ret["obs"][key] = ret["obs"][key][:,:,:3,:,:] # Take the first 3 channel
                # if "state" in key: # We remove velocity from the state
                #     if isinstance(ret["obs"][key], torch.Tensor):
                #         ret["obs"][key] = torch.cat([ret["obs"][key][...,:9], ret["obs"][key][...,18:]], axis=-1)
                #     elif isinstance(ret["obs"][key], np.ndarray):
                #         ret["obs"][key] = np.concatenate([ret["obs"][key][...,:9], ret["obs"][key][...,18:]], axis=-1)
                #     else:
                #         raise NotImplementedError()
        # if "keyframe_states" in ret.keys():
        #     # We remove velocity from the state
        #     if isinstance(ret["keyframe_states"], torch.Tensor):
        #         ret["keyframe_states"] = torch.cat([ret["keyframe_states"][...,:9], ret["keyframe_states"][...,18:]], axis=-1)
        #     elif isinstance(ret["keyframe_states"], np.ndarray):
        #         ret["keyframe_states"] = np.concatenate([ret["keyframe_states"][...,:9], ret["keyframe_states"][...,18:]], axis=-1)
        #     else:
        #         raise NotImplementedError()
        if self.horizon > 1:
            batch_flat_idx = [i for i in range(batch_size) for j in range(self.horizon-ret_len[i])]
            ret_flat_idx = [j for i in range(batch_size) for j in range(self.horizon-ret_len[i])]
            # print(ret_len, batch_flat_idx, ret_flat_idx)
            # for key in ret.keys():
            #     if "keyframe" in key or "keytime" in key:
            #         ret[key] = ret[key][:, -1] # We only take the last step of the horizon since we want to train the key frame model
            if "actions" in ret.keys(): # Set zero actions
                ret["actions"][batch_flat_idx,ret_flat_idx,:] = 0
            # for i in range(len(batch_idx)):
            #     if self.horizon-ret_len[i]:
                    # if "obs" in ret.keys(): # Concat obs
                    #     for key in ret["obs"].keys():
                    #         if isinstance(ret["obs"][key], (list,tuple)):
                    #             ret["obs"][key] = ret["obs"][key][0]
                    #         supp = np.array([ret["obs"][key][0][0],]*(self.horizon-ret_len[i]))
                    #         ret["obs"][key][i] = np.concatenate([supp, ret["obs"][key][i][-ret_len[i]:]], axis=0)
                    # if "actions" in ret.keys(): # Set zero actions
                            # supp = np.array([0*np.zeros(ret["actions"].shape[-1]),]*(self.horizon-ret_len[i]))
                            # ret["actions"][i] = np.concatenate([supp, ret["actions"][i][-ret_len[i]:]], axis=0)
        ret["is_valid"] = is_valid

        if keyframe_type == "bc":
            ret["keyframe_states"] = ret["keyframe_states"][:,:,0] # Only keep the first state and tcp pose
            ret["keytime_differences"] = ret["keytime_differences"][:,:,0]
            ret["keyframe_actions"] = ret["keyframe_actions"][:,:,0]
            ret["keyframe_masks"] = ret["keyframe_masks"][:,:,0]

        if device is not None:
            ret = ret.to_torch(device=device, dtype="float32", non_blocking=True)

        ret["normed_states"] = copy.deepcopy(ret["obs"]["state"])
        if action_normalizer is not None:
            # for key in ["actions", "keyframe_actions"]:
            #     if key in ret:
            #         ret[key] = action_normalizer.normalize(ret[key])
            ret["normed_actions"] = action_normalizer.normalize(ret["actions"])
        elif obsact_normalizer is not None:
            if "actions" in ret and "obs" in ret:
                action_dim = ret["actions"].shape[-1]
                data = torch.cat([ret["obs"]["state"], ret["actions"]], dim=-1)
                data = obsact_normalizer.normalize(data)
                ret["normed_states"] = data[...,:-action_dim]
                ret["normed_actions"] = data[...,-action_dim:]
            # if "keyframe_states" in ret and "keyframe_actions" in ret: # We do not norm for keyframe prediction
            #     data = torch.cat([ret["keyframe_states"], ret["keyframe_actions"]], dim=-1)
            #     data = obsact_normalizer.normalize(data)
            #     ret["keyframe_states"] = data[...,:-action_dim]
            #     ret["keyframe_actions"] = data[...,-action_dim:]
            # if "ep_first_obs" in ret: # We do not consider ep first obs for now
            #     data = torch.cat([ret['ep_first_obs']['state'][:,0], ret["actions"][:,0]], dim=-1)
            #     data = obsact_normalizer.normalize(data)
            #     ret['ep_first_obs']['state'] = data[...,:-action_dim]
            #     for key in ret['ep_first_obs']:
            #         if key != "state":
            #             ret['ep_first_obs'][key] = ret['ep_first_obs'][key][:,0]

        if (obs_mask is not None) and ("obs" in ret.keys()):
            obs_mask = obs_mask.cpu().numpy()
            for key in ret["obs"].keys():
                ret["obs"][key] = ret["obs"][key][:,obs_mask,...]

        if mode=="eval":
            return ret
        data_queue.append(ret)
        self.thread_count -= 1

    def sample(self, batch_size, auto_restart=True, drop_last=True, device=None, obs_mask=None, require_mask=False, action_normalizer=None, obsact_normalizer=None, mode="train", whole_traj=False, keyframe_type="gpt"):
        if mode=="eval":
            return self.pre_fetch(batch_size,auto_restart,drop_last,device,obs_mask,action_normalizer,obsact_normalizer, mode=mode, whole_traj=whole_traj, keyframe_type=keyframe_type)

        ret = self.prefetched_data(whole_traj)
        if ret is not None:
            if self.thread_count < self.max_threads:
                self.thread_count += 1
                # _thread.start_new_thread(self.pre_fetch, (batch_size,auto_restart,drop_last,device,obs_mask,action_normalizer,mode,whole_traj))
                new_thread = threading.Thread(target=self.pre_fetch, args=(batch_size,auto_restart,drop_last,device,obs_mask,action_normalizer,obsact_normalizer,mode,whole_traj,keyframe_type))
                new_thread.setDaemon(True)
                new_thread.start()
            return ret

        self.pre_fetch(batch_size,auto_restart,drop_last,device,obs_mask,action_normalizer,obsact_normalizer,mode,whole_traj,keyframe_type)
        ret = self.prefetched_data(whole_traj)
        if (obs_mask is not None) or (not require_mask): # If we don't need mask or the mask is provided, we can pre-fetch the next batch
            if self.thread_count < self.max_threads:
                self.thread_count += 1
                # _thread.start_new_thread(self.pre_fetch, (batch_size,auto_restart,drop_last,device,obs_mask,action_normalizer,mode,whole_traj))
                new_thread = threading.Thread(target=self.pre_fetch, args=(batch_size,auto_restart,drop_last,device,obs_mask,action_normalizer,obsact_normalizer,mode,whole_traj,keyframe_type))
                new_thread.setDaemon(True)
                new_thread.start()
        return ret

    def mini_batch_sampler(self, batch_size, drop_last=False, auto_restart=False, max_num_batches=-1):
        if self.sampling is not None:
            old_replacement = self.sampling.with_replacement
            self.sampling.with_replacement = False
            self.sampling.restart()
        for i in count(1):
            if i > max_num_batches and max_num_batches != -1:
                break
            items = self.sample(batch_size, auto_restart, drop_last)
            if items is None:
                self.sampling.with_replacement = old_replacement
                break
            yield items

    def close(self):
        if self.file_loader is not None:
            self.file_loader.close()

    def __del__(self):
        self.close()
