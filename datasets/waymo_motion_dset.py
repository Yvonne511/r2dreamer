from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader


MAP_REWARD_KEYS = (
    "reward_lane_alignment",
    "reward_lane_center_offset",
    "reward_collision",
    "reward_road_boundary",
)


class WaymoMotionDataset(Dataset):
    def __init__(
        self,
        path,
        n_rollout=None,
        image_size=None,
        window_size=20,
        frame_skip=1,
        **kwargs,
    ):
        self.path = Path(path)
        self.obs_dir = self.path / "obses"
        self.window_size = int(window_size) + 1
        self.frame_skip = int(frame_skip)
        self.image_size = _normalize_image_size(image_size)

        self.actions = torch.load(self.path / "actions.pth", map_location="cpu").numpy().astype(np.float32, copy=False)
        self.rewards = torch.load(self.path / "rewards.pth", map_location="cpu").numpy().astype(np.float32, copy=False)
        self.seq_lengths = torch.load(self.path / "seq_lengths.pth", map_location="cpu").numpy().astype(
            np.int64, copy=False
        )
        with open(self.path / "scenario_ids.json", "r", encoding="utf-8") as file:
            self.segment_paths = list(json.load(file))

        if n_rollout is not None:
            n_rollout = int(n_rollout)
            self.actions = self.actions[:n_rollout]
            self.rewards = self.rewards[:n_rollout]
            self.seq_lengths = self.seq_lengths[:n_rollout]
            self.segment_paths = self.segment_paths[:n_rollout]

        num_rollouts = len(self.segment_paths)
        if not (len(self.actions) == len(self.rewards) == len(self.seq_lengths) == num_rollouts):
            raise ValueError("Waymo motion dataset files have inconsistent rollout counts.")

        self.video_paths = [self.obs_dir / f"{idx}.mp4" for idx in range(num_rollouts)]
        self.bev_video_paths = None
        if (self.path / "bev_obses").is_dir():
            self.bev_video_paths = [self.path / "bev_obses" / f"{idx}.mp4" for idx in range(num_rollouts)]
        missing = [str(path) for path in self.video_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing rollout videos: {missing[:4]}")

        self.vehicle_poses = [(np.arange(int(length), dtype=np.int64), None) for length in self.seq_lengths.tolist()]
        self.slices = []
        for episode_idx, length in enumerate(self.seq_lengths.tolist()):
            for start in range(0, length, self.frame_skip):
                end = start + self.window_size * self.frame_skip
                if end > length:
                    break
                self.slices.append((episode_idx, start, end))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        episode_idx, start, end = self.slices[index]
        episode_length = int(self.seq_lengths[episode_idx])
        step_ids = np.arange(start, end, self.frame_skip, dtype=np.int64)

        reader = VideoReader(str(self.video_paths[episode_idx]), num_threads=1)
        images = reader.get_batch(step_ids).asnumpy()

        if self.bev_video_paths is not None:
            bev_reader = VideoReader(str(self.bev_video_paths[episode_idx]), num_threads=1)
            bev_images = bev_reader.get_batch(step_ids).asnumpy()
        actions = self.actions[episode_idx, start:end].reshape(-1, self.frame_skip * self.actions.shape[-1])

        raw_rewards = self.rewards[episode_idx, start:end:self.frame_skip]
        rewards = np.zeros((len(step_ids), 1), dtype=np.float32)
        if len(step_ids) > 1:
            rewards[1:, 0] = -(raw_rewards[:-1, 2] + raw_rewards[:-1, 3])

        is_first = np.zeros((len(step_ids), 1), dtype=bool)
        is_last = np.zeros((len(step_ids), 1), dtype=bool)
        is_terminal = np.zeros((len(step_ids), 1), dtype=bool)
        if start == 0:
            is_first[0, 0] = True
        if end == episode_length:
            is_last[-1, 0] = True
            is_terminal[-1, 0] = True

        episode = {
            "image": images,
            **({"bev_image": bev_images} if self.bev_video_paths is not None else {}),
            "action": actions,
            "reward": rewards,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "episode": np.full((len(step_ids),), episode_idx, dtype=np.int64),
            "step": step_ids,
        }
        for reward_idx, key in enumerate(MAP_REWARD_KEYS):
            component = np.zeros((len(step_ids), 1), dtype=np.float32)
            if len(step_ids) > 1:
                component[1:, 0] = raw_rewards[:-1, reward_idx]
            episode[key] = component
        return episode

def _normalize_image_size(value):
    if value is None:
        return None
    return tuple(value)
