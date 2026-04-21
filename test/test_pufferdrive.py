import os
from pathlib import Path

import cv2
import numpy as np
import yaml
import sys


puffer_root = "/scratch/yw4142/PufferDrive"
map_dir = "/scratch/yw4142/datasets/ad/WOMD/resources/drive/binaries/training"
puffer_config_yaml = "/home/yw4142/ad/r2dreamer/configs/env/puffer_drive.yaml"
render_output_dir = Path('/scratch/yw4142/checkpoints/ad/LatentDrive/pufferdrive_test_renders')
seed = 0
steps = 300

sys.path.insert(0, str(puffer_root))
from pufferlib.pufferl import load_env
from pufferlib.ocean.drive.drive import RenderView


def get_video_writer(
	writers: dict[Path, cv2.VideoWriter],
	output_path: Path,
	frame_shape: tuple[int, int, int],
	fps: int = 10,
) -> cv2.VideoWriter:
	writer = writers.get(output_path)
	if writer is not None:
		return writer

	height, width, _ = frame_shape
	writer = cv2.VideoWriter(
		str(output_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		float(fps),
		(width, height),
	)
	if not writer.isOpened():
		raise RuntimeError(f"Failed to open video writer for {output_path}")

	writers[output_path] = writer
	return writer

def load_pufferdrive_vecenv():
	# Preserve original structure:
	# args = load_config("puffer_drive", config_dir="...")
	# args["env"]["map_dir"] = "..."
	# args["vec"] = {...}
	# vecenv = load_env("puffer_drive", args)
	with open(puffer_config_yaml, "r", encoding="utf-8") as f:
		yaml_cfg = yaml.safe_load(f) or {}

	cfg = yaml_cfg
	cfg["env"]["num_agents"] = 2
	cfg["env"]["map_dir"] = map_dir
	cfg["vec"]["backend"] = "Serial"
	cfg["env"]["control_mode"] = "control_sdc_only"
	cfg["env"]["dynamics_model"] = "delta"
	cfg["env"]["action_type"] = "continuous"
	cfg["env"]["render_mode"] = 1
	# cfg["env"]["num_agents"] = args.num_agents
	# cfg["env"]["num_maps"] = args.num_maps
	# cfg["vec"] = make_vec_config(mode, args)
	vecenv = load_env("puffer_drive", cfg)
	return vecenv, cfg

def rollout():
	os.chdir(puffer_root)

	vecenv, cfg = load_pufferdrive_vecenv()
	driver_env = vecenv.driver_env
	render_output_dir.mkdir(parents=True, exist_ok=True)
	video_writers: dict[Path, cv2.VideoWriter] = {}
	scenario_ids = driver_env.scenario_ids

	vecenv.reset(seed=seed) # number_agent = num_agents * batch_size
	reward_sums = []
	env_count = 0

	try:
		for step_idx in range(steps):
			action = vecenv.action_space.sample()
			expert_actions, mask = vecenv.get_expert_actions()
			expert_actions = expert_actions.reshape(vecenv.action_space.shape)
			obs, rewards, terminals, truncations, info = vecenv.step(expert_actions)
			for env_id in range(driver_env.num_envs):
				rc = driver_env.get_reward_components(env_id)
				n = driver_env.agent_offsets[env_id + 1] - driver_env.agent_offsets[env_id]
				assert all(v.shape == (n,) for v in rc.values()), \
					f"step {step_idx} env {env_id}: unexpected shapes {({k: v.shape for k, v in rc.items()})}"
			for env_id in range(driver_env.num_envs):
				scenario_id = scenario_ids[env_id].rstrip("\x00")
				for view_mode, suffix in [(RenderView.AGENT_PERSP, ""), (RenderView.BEV_AGENT_OBS, "_bev")]:
					images, batch_indices = driver_env.render_all_controlled_agent_views(env_id=env_id, view_mode=view_mode)
					for local_agent_idx, (image, batch_idx) in enumerate(zip(images, batch_indices)):
						output_path = render_output_dir / (
							f"{scenario_id}_env_{env_id:03d}_agent_{local_agent_idx:03d}_batch_{int(batch_idx):05d}{suffix}.mp4"
						)
						writer = get_video_writer(video_writers, output_path, image.shape)
						writer.write(cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR))
			
			reward_sums.append(float(np.asarray(rewards).sum()))
	finally:
		for writer in video_writers.values():
			writer.release()

	for i, env in enumerate(vecenv.envs):
		print(len(env.env_ids))
		env_count += len(env.env_ids)
	print(f"Total agents_per_batch", {vecenv.agents_per_batch})
	print(f"Action Space", {vecenv.action_space.sample().shape})
	print(f"Total environments: {env_count}")
	print(f"Saved renders to: {render_output_dir}")
		

	vecenv.close()
	return reward_sums


if __name__ == "__main__":
	rollout()
