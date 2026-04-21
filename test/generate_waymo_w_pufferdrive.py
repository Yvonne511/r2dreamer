from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm


PUFFER_ROOT = Path("/scratch/yw4142/PufferDrive")
MAP_DIR = Path("/scratch/yw4142/datasets/ad/WOMD/resources/drive/binaries/training")
PUFFER_CONFIG_YAML = Path("/home/yw4142/ad/r2dreamer/configs/env/puffer_drive.yaml")
OUTPUT_ROOT = Path("/scratch/yw4142/datasets/ad/waymo_pufferdrive_256")

SCENES_PER_BATCH = 64
SEED = 0
START_MAP_ID = 0
END_MAP_ID = None
MAX_MAPS = None  # set to None for full dataset
OVERWRITE = True
FPS = 10


MAP_FILE_RE = re.compile(r"map_(\d+)\.bin$")

sys.path.insert(0, str(PUFFER_ROOT))
from pufferlib.ocean.drive.drive import Drive, RenderView


def discover_available_map_ids(map_dir: Path) -> list[int]:
    map_ids: list[int] = []
    for path in sorted(map_dir.glob("map_*.bin")):
        match = MAP_FILE_RE.match(path.name)
        if match is None:
            continue
        map_ids.append(int(match.group(1)))
    if not map_ids:
        raise FileNotFoundError(f"No map_*.bin files found under {map_dir}")
    return map_ids


def get_selected_map_ids(available_map_ids: list[int]) -> list[int]:
    end_map_id = len(available_map_ids) if END_MAP_ID is None else int(END_MAP_ID)
    selected = [map_id for map_id in available_map_ids if START_MAP_ID <= map_id < end_map_id]
    if MAX_MAPS is not None:
        selected = selected[: int(MAX_MAPS)]
    if not selected:
        raise ValueError("No map ids selected. Check START_MAP_ID, END_MAP_ID, and MAX_MAPS.")
    return selected


def iter_chunks(items: list[int], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError(f"SCENES_PER_BATCH must be positive, got {chunk_size}")
    total_chunks = (len(items) + chunk_size - 1) // chunk_size
    for start in tqdm(
        range(0, len(items), chunk_size),
        total=total_chunks,
        desc="Exporting chunks",
        unit="chunk",
    ):
        yield start, items[start : start + chunk_size]


def prepare_output_dirs(output_root: Path) -> Path:
    if output_root.exists():
        if not OVERWRITE:
            raise FileExistsError(f"{output_root} already exists. Set OVERWRITE = True to replace it.")
        shutil.rmtree(output_root)
    obses_dir = output_root / "obses"
    obses_dir.mkdir(parents=True, exist_ok=True)
    return obses_dir


def get_video_writer(
    writers: dict[int, cv2.VideoWriter],
    trajectory_index: int,
    output_path: Path,
    frame_shape: tuple[int, int, int],
) -> cv2.VideoWriter:
    writer = writers.get(trajectory_index)
    if writer is not None:
        return writer

    height, width, _ = frame_shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    writers[trajectory_index] = writer
    return writer


def load_pufferdrive_env(fixed_map_ids: list[int], available_map_count: int):
    with open(PUFFER_CONFIG_YAML, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    cfg["env"]["map_dir"] = str(MAP_DIR)
    cfg["env"]["num_maps"] = int(available_map_count)
    cfg["env"]["num_agents"] = len(fixed_map_ids)
    cfg["env"]["control_mode"] = "control_sdc_only"
    cfg["env"]["dynamics_model"] = "delta"
    cfg["env"]["action_type"] = "continuous"
    cfg["env"]["render_mode"] = 1
    cfg["env"]["resample_frequency"] = 0
    cfg["env"]["fixed_map_ids"] = list(fixed_map_ids)

    os.chdir(PUFFER_ROOT)
    env = Drive(**cfg["env"])
    return env, cfg


def get_controlled_batch_indices(driver_env: Drive) -> np.ndarray:
    agent_offsets = np.asarray(driver_env.agent_offsets, dtype=np.int32)
    if agent_offsets.shape[0] != driver_env.num_envs + 1:
        raise RuntimeError(
            f"Expected agent_offsets shape {(driver_env.num_envs + 1,)}, got {agent_offsets.shape}"
        )

    controlled_counts = np.diff(agent_offsets)
    if not np.all(controlled_counts == 1):
        raise RuntimeError(
            "Expected exactly one controlled agent per env in control_sdc_only mode, "
            f"got counts {controlled_counts.tolist()}"
        )

    return agent_offsets[:-1]


def stack_state_snapshot(state_dict: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            state_dict["x"],
            state_dict["y"],
            state_dict["z"],
            state_dict["heading"],
            state_dict["id"].astype(np.float32, copy=False),
            state_dict["length"],
            state_dict["width"],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def export_chunk(
    fixed_map_ids: list[int],
    global_start_index: int,
    obses_dir: Path,
    available_map_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    driver_env, cfg = load_pufferdrive_env(fixed_map_ids=fixed_map_ids, available_map_count=available_map_count)

    batch_size = len(fixed_map_ids)
    t_max = int(cfg["env"]["episode_length"]) - 1
    actions = np.zeros((batch_size, t_max, 3), dtype=np.float32)
    rewards = np.zeros((batch_size, t_max, 4), dtype=np.float32)
    states = np.zeros((batch_size, t_max, 7), dtype=np.float32)
    seq_lengths = np.zeros(batch_size, dtype=np.int64)
    scenario_ids_by_agent = [""] * batch_size
    seen_invalid = np.zeros(batch_size, dtype=bool)
    video_writers: dict[int, cv2.VideoWriter] = {}

    if driver_env.num_envs != batch_size:
        raise RuntimeError(
            f"Expected one env per map in control_sdc_only mode, got num_envs={driver_env.num_envs} for batch_size={batch_size}"
        )

    driver_env.reset(seed=SEED)
    controlled_batch_indices = get_controlled_batch_indices(driver_env)
    scenario_ids = driver_env.scenario_ids
    for env_id, batch_idx in enumerate(controlled_batch_indices.tolist()):
        scenario_ids_by_agent[batch_idx] = scenario_ids[env_id]

    try:
        for step_idx in range(t_max):
            state_snapshot = stack_state_snapshot(driver_env.get_global_agent_state())
            expert_actions, valid_mask = driver_env.get_expert_actions()
            expert_actions = np.asarray(expert_actions, dtype=np.float32)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if expert_actions.shape != (batch_size, 3):
                raise RuntimeError(f"Expected expert action shape {(batch_size, 3)}, got {expert_actions.shape}")

            reactivated = seen_invalid & valid_mask
            if np.any(reactivated):
                bad_indices = np.flatnonzero(reactivated).tolist()
                raise RuntimeError(f"Expert validity became true again after ending for trajectories {bad_indices}")

            for env_id, batch_idx in enumerate(controlled_batch_indices.tolist()):
                if not valid_mask[batch_idx]:
                    continue

                image = driver_env.render(
                    view_mode=RenderView.AGENT_PERSP,
                    draw_traces=True,
                    env_id=env_id,
                    agent_idx=0,
                    return_rgb=True,
                )

                trajectory_index = global_start_index + batch_idx
                output_path = obses_dir / f"{trajectory_index}.mp4"
                writer = get_video_writer(video_writers, trajectory_index, output_path, image.shape)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))

            driver_env.step(expert_actions)
            rc = driver_env.get_reward_components()
            step_rewards = np.stack(
                [rc["alignment_angle"], rc["distance_to_center"], rc["collision"], rc["offroad"]], axis=-1
            )

            states[valid_mask, step_idx] = state_snapshot[valid_mask]
            actions[valid_mask, step_idx] = expert_actions[valid_mask]
            rewards[valid_mask, step_idx] = step_rewards[valid_mask]
            seq_lengths += valid_mask.astype(np.int64)
            seen_invalid |= ~valid_mask
    finally:
        for writer in video_writers.values():
            writer.release()
        # Repeated render + explicit close currently triggers a native double-free in this path.

    if np.any(seq_lengths == 0):
        zero_length_indices = np.flatnonzero(seq_lengths == 0).tolist()
        raise RuntimeError(f"Trajectories with zero valid steps encountered: {zero_length_indices}")
    if any(not scenario_id for scenario_id in scenario_ids_by_agent):
        raise RuntimeError("Missing scenario ids for one or more exported trajectories")

    return actions, rewards, states, seq_lengths, scenario_ids_by_agent


def export_dataset():
    available_map_ids = discover_available_map_ids(MAP_DIR)
    selected_map_ids = get_selected_map_ids(available_map_ids)
    available_map_count = len(available_map_ids)
    obses_dir = prepare_output_dirs(OUTPUT_ROOT)

    all_actions = []
    all_rewards = []
    all_states = []
    all_seq_lengths = []
    all_scenario_ids: list[str] = []

    print(
        f"Exporting {len(selected_map_ids)} maps from {MAP_DIR} "
        f"to {OUTPUT_ROOT} with chunk size {SCENES_PER_BATCH}"
    )

    for global_start_index, map_id_chunk in iter_chunks(selected_map_ids, SCENES_PER_BATCH):
        print(
            f"Chunk starting at dataset row {global_start_index}: "
            f"map ids {map_id_chunk[0]}..{map_id_chunk[-1]} ({len(map_id_chunk)} scenes)"
        )
        chunk_actions, chunk_rewards, chunk_states, chunk_seq_lengths, chunk_scenario_ids = export_chunk(
            fixed_map_ids=map_id_chunk,
            global_start_index=global_start_index,
            obses_dir=obses_dir,
            available_map_count=available_map_count,
        )
        all_actions.append(chunk_actions)
        all_rewards.append(chunk_rewards)
        all_states.append(chunk_states)
        all_seq_lengths.append(chunk_seq_lengths)
        all_scenario_ids.extend(chunk_scenario_ids)

    actions_tensor = torch.from_numpy(np.concatenate(all_actions, axis=0))
    rewards_tensor = torch.from_numpy(np.concatenate(all_rewards, axis=0))
    states_tensor = torch.from_numpy(np.concatenate(all_states, axis=0))
    seq_lengths_tensor = torch.from_numpy(np.concatenate(all_seq_lengths, axis=0))

    N, T = len(all_scenario_ids), actions_tensor.shape[1]
    assert actions_tensor.shape == (N, T, 3), f"actions {actions_tensor.shape} != ({N}, {T}, 3)"
    assert rewards_tensor.shape == (N, T, 4), f"rewards {rewards_tensor.shape} != ({N}, {T}, 4)"
    assert states_tensor.shape == (N, T, 7), f"states {states_tensor.shape} != ({N}, {T}, 7)"
    assert seq_lengths_tensor.shape == (N,), f"seq_lengths {seq_lengths_tensor.shape} != ({N},)"
    assert len(list((OUTPUT_ROOT / "obses").glob("*.mp4"))) == N, f"expected {N} videos in obses/"

    torch.save(actions_tensor, OUTPUT_ROOT / "actions.pth")
    torch.save(rewards_tensor, OUTPUT_ROOT / "rewards.pth")
    torch.save(states_tensor, OUTPUT_ROOT / "states.pth")
    torch.save(seq_lengths_tensor, OUTPUT_ROOT / "seq_lengths.pth")

    with open(OUTPUT_ROOT / "scenario_ids.json", "w", encoding="utf-8") as file:
        json.dump(all_scenario_ids, file, indent=2)

    print(
        f"Saved {len(all_scenario_ids)} trajectories with padded horizon {actions_tensor.shape[1]} "
        f"under {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    export_dataset()
