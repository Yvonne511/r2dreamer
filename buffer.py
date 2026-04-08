import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        self._buffer.extend(data.unsqueeze(1))

    def add_episode(self, data):
        # This is a single episode TensorDict with batch size (T, ...).
        # Lift it to (1, T, ...) so ReplayBuffer keeps episode structure.
        self._buffer.extend(data.unsqueeze(0))

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()

class OfflineDatasetBuffer:

    def __init__(self, config, dataset):
        self.device = torch.device(config.buffer.device)
        self.storage_device = torch.device(config.buffer.storage_device)
        self.batch_size = int(config.buffer.batch_size)
        self.batch_length = int(config.buffer.batch_length)
        self.num_eps = 0
        self.dataset = dataset
        assert dataset.window_size == self.batch_length + 1, (f"dataset.window_size must equal batch_length + 1 ")
        num_workers = int(getattr(config.buffer, "num_workers", 4))
        pin_memory = bool(getattr(config.buffer, "pin_memory", self.device.type == "cuda"))
        persistent_workers = bool(getattr(config.buffer, "persistent_workers", num_workers > 0))
        self.dataloader = DataLoader( # TODP: pass num_workers, pin_memory, etc. from config
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        self.iterator = iter(self.dataloader)
        S, K = config.model.rssm.stoch, config.model.rssm.discrete
        D = config.model.rssm.deter
        self.stoch_cache = []
        self.deter_cache = []
        self.valid_cache = []
        for segment_index, _segment_id in enumerate(dataset.segment_paths):
            n = len(dataset.vehicle_poses[segment_index][0])
            self.stoch_cache.append(torch.zeros(n, S, K, dtype=torch.float32, device=self.storage_device))
            self.deter_cache.append(torch.zeros(n, D, dtype=torch.float32, device=self.storage_device))
            self.valid_cache.append(torch.zeros(n, dtype=torch.bool, device=self.storage_device))

    def _next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def add_transition(self, data):
        raise NotImplementedError("Unused for dataset-backed offline buffer.")

    def add_episode(self, data):
        raise NotImplementedError("Unused for dataset-backed offline buffer.")

    def sample(self):
        batch = self._next_batch()
        B, Tp1 = batch["action"].shape[:2]

        assert B == self.batch_size
        assert Tp1 == self.batch_length + 1

        # read initial latent from first context frame of each window
        ep0 = batch["episode"][:, 0].to(torch.long)
        t0 = batch["step"][:, 0].to(torch.long)

        init_stoch = torch.stack(
            [self.stoch_cache[e][t] for e, t in zip(ep0.tolist(), t0.tolist())], dim=0
        )
        init_deter = torch.stack(
            [self.deter_cache[e][t] for e, t in zip(ep0.tolist(), t0.tolist())], dim=0
        )

        initial = (
            init_stoch.to(self.device, non_blocking=True),
            init_deter.to(self.device, non_blocking=True),
        )

        # move batch to target device
        td = TensorDict(
            {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()},
            batch_size=[B, Tp1],
            device=self.device,
        )

        data = td[:, 1:].clone()
        data.set_("action", td["action"][:, :-1])  # same alignment as original

        # identity for update(): positions corresponding to data[:, 0:T]
        index = {
            "episode": td["episode"][:, 1:].to(torch.long),
            "step": td["step"][:, 1:].to(torch.long),
        }

        return data, index, initial

    def update(self, index, stoch, deter):
        ep = index["episode"].detach().to(self.storage_device)
        step = index["step"].detach().to(self.storage_device)
        stoch = stoch.detach().to(self.storage_device)
        deter = deter.detach().to(self.storage_device)

        B, T = ep.shape
        for b in range(B):
            e = int(ep[b, 0].item())  # all steps in a window are same episode
            ts = step[b]              # (T,)
            self.stoch_cache[e][ts] = stoch[b]
            self.deter_cache[e][ts] = deter[b]
            self.valid_cache[e][ts] = True

    def count(self):
        return len(self.dataset)
