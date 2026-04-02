import pathlib

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


def make_dummy_envs(config, replay_buffer):
    episodes, metadata = _load_episodes(config.dataset)
    if not episodes:
        raise ValueError(f"No expert episodes found in {config.dataset.path}")

    obs_space = _build_observation_space(episodes[0])
    act_space = _build_action_space(episodes[0], config.dataset.action_space, metadata)
    stoch_shape = (int(config.model.rssm.stoch), int(config.model.rssm.discrete))
    deter_size = int(config.model.rssm.deter)

    total_steps = 0
    for episode_id, episode in enumerate(episodes):
        td = _episode_to_tensordict(episode, episode_id, stoch_shape, deter_size, act_space, metadata)
        replay_buffer.add_episode(td)
        total_steps += td.shape[0]

    print(f"Loaded {len(episodes)} expert episodes ({total_steps} transitions) from {config.dataset.path}")
    return None, None, obs_space, act_space


def _load_episodes(dataset_config):
    path = pathlib.Path(dataset_config.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    fmt = str(dataset_config.format)
    if fmt == "auto":
        fmt = path.suffix.lower().lstrip(".")

    if fmt in ("pt", "pth"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    elif fmt == "npz":
        with np.load(path, allow_pickle=True) as npz:
            payload = {key: npz[key] for key in npz.files}
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")

    return _normalize_payload(payload)


def _normalize_payload(payload):
    metadata = {}
    if isinstance(payload, dict):
        metadata = _normalize_metadata(payload.get("metadata", payload.get("action_space", {})))
        if "episodes" in payload:
            return [_normalize_episode(ep) for ep in payload["episodes"]], metadata
        stacked = {k: v for k, v in payload.items() if k not in ("metadata", "action_space")}
        if _is_stacked_episode_dict(stacked):
            return _split_stacked_episodes(stacked), metadata
    elif isinstance(payload, (list, tuple)):
        return [_normalize_episode(ep) for ep in payload], metadata
    raise ValueError("Unsupported expert dataset payload. Expected a list of episodes, an 'episodes' dict, or stacked arrays.")


def _is_stacked_episode_dict(payload):
    if not isinstance(payload, dict) or not payload:
        return False
    first_dims = []
    for value in payload.values():
        tensor = _to_tensor(value)
        if tensor.ndim < 2:
            return False
        first_dims.append(tensor.shape[0])
    return len(set(first_dims)) == 1 and "action" in payload


def _split_stacked_episodes(payload):
    tensors = {key: _to_tensor(value) for key, value in payload.items()}
    episode_num = next(iter(tensors.values())).shape[0]
    episodes = []
    for index in range(episode_num):
        episode = {key: value[index] for key, value in tensors.items()}
        episodes.append(_normalize_episode(episode))
    return episodes


def _normalize_episode(episode):
    if not isinstance(episode, dict):
        raise ValueError("Each expert episode must be a dict of arrays or tensors.")
    out = {key: _to_tensor(value) for key, value in episode.items()}
    if "action" not in out:
        raise ValueError("Expert episode is missing required key 'action'.")

    time_dim = out["action"].shape[0]
    if "reward" not in out:
        out["reward"] = torch.zeros(time_dim, 1, dtype=torch.float32)
    else:
        out["reward"] = _ensure_column(out["reward"], torch.float32)

    out["is_first"] = _ensure_flag(out.get("is_first"), time_dim, first=True, last=False)
    out["is_last"] = _ensure_flag(out.get("is_last"), time_dim, first=False, last=True)
    out["is_terminal"] = _ensure_flag(out.get("is_terminal"), time_dim, first=False, last=True)
    return out


def _build_observation_space(episode):
    spaces = {}
    ignored = {"action", "reward", "episode", "stoch", "deter"}
    for key, value in episode.items():
        if key in ignored:
            continue
        shape = tuple(value.shape[1:])
        if value.dtype == torch.uint8:
            spaces[key] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        elif value.dtype == torch.bool:
            spaces[key] = gym.spaces.Box(0, 1, shape=shape, dtype=bool)
        else:
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
    return gym.spaces.Dict(spaces)


def _build_action_space(episode, action_cfg, metadata):
    action = _to_tensor(episode["action"])
    kind = str(getattr(action_cfg, "kind", "auto"))
    if kind == "auto":
        kind = str(metadata.get("kind", "continuous"))
    shape = getattr(action_cfg, "shape", None)
    resolved_shape = tuple(shape) if shape is not None else tuple(action.shape[1:])

    if kind == "discrete":
        if not resolved_shape:
            classes = int(action.max().item()) + 1 if action.ndim == 1 else int(action.shape[-1])
            resolved_shape = (classes,)
        space = gym.spaces.Box(low=0.0, high=1.0, shape=resolved_shape, dtype=np.float32)
        space.discrete = True
        return space
    if kind == "multi_discrete":
        if not resolved_shape:
            raise ValueError("dataset.action_space.shape must be set for multi_discrete offline data.")
        space = gym.spaces.Box(low=0.0, high=1.0, shape=resolved_shape, dtype=np.float32)
        space.multi_discrete = True
        return space

    low = np.broadcast_to(np.asarray(action_cfg.low, dtype=np.float32), resolved_shape)
    high = np.broadcast_to(np.asarray(action_cfg.high, dtype=np.float32), resolved_shape)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def _episode_to_tensordict(episode, episode_id, stoch_shape, deter_size, act_space, metadata):
    episode = {key: value.clone() for key, value in episode.items()}
    episode["action"] = _format_action(episode["action"], act_space, metadata)
    time_dim = episode["action"].shape[0]
    episode["episode"] = torch.full((time_dim,), episode_id, dtype=torch.int32)
    if "stoch" not in episode:
        episode["stoch"] = torch.zeros(time_dim, *stoch_shape, dtype=torch.float32)
    else:
        episode["stoch"] = _to_tensor(episode["stoch"]).to(torch.float32)
    if "deter" not in episode:
        episode["deter"] = torch.zeros(time_dim, deter_size, dtype=torch.float32)
    else:
        episode["deter"] = _to_tensor(episode["deter"]).to(torch.float32)
    return TensorDict(episode, batch_size=(time_dim,), device="cpu")


def _format_action(action, act_space, metadata):
    action = _to_tensor(action)
    if getattr(act_space, "discrete", False):
        classes = act_space.shape[0]
        if action.ndim == 1 or (action.ndim == 2 and action.shape[-1] == 1):
            action = torch.nn.functional.one_hot(action.long().reshape(-1), num_classes=classes)
        return action.to(torch.float32)
    return action.to(torch.float32)


def _ensure_column(value, dtype):
    value = _to_tensor(value).to(dtype)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    return value


def _ensure_flag(value, time_dim, first=False, last=False):
    if value is None:
        value = torch.zeros(time_dim, 1, dtype=torch.bool)
        if first:
            value[0] = True
        if last:
            value[-1] = True
        return value
    value = _to_tensor(value).to(torch.bool)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    return value


def _to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.asarray(value))


def _normalize_metadata(metadata):
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, np.ndarray) and metadata.shape == ():
        item = metadata.item()
        if isinstance(item, dict):
            return item
    return {}
