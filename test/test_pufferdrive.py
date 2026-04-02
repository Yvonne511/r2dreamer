import os

import numpy as np
import yaml
import sys


puffer_root = "/scratch/yw4142/PufferDrive"
map_dir = "/scratch/yw4142/datasets/ad/WOMD/resources/drive/binaries/training"
puffer_config_yaml = "/home/yw4142/ad/dreamerv3/conf/env/puffer_drive.yaml"
seed = 0
steps = 100

# os.chdir(puffer_root)
sys.path.insert(0, str(puffer_root))
from pufferlib.pufferl import load_env

def load_pufferdrive_vecenv():
	# Preserve original structure:
	# args = load_config("puffer_drive", config_dir="...")
	# args["env"]["map_dir"] = "..."
	# args["vec"] = {...}
	# vecenv = load_env("puffer_drive", args)
	with open(puffer_config_yaml, "r", encoding="utf-8") as f:
		yaml_cfg = yaml.safe_load(f) or {}

	cfg = yaml_cfg
	cfg["env"]["map_dir"] = map_dir
	cfg["vec"]["backend"] = "Serial"
	cfg["env"]["control_mode"] = "control_wosac"
	# cfg["env"]["num_agents"] = args.num_agents
	# cfg["env"]["num_maps"] = args.num_maps
	# cfg["env"]["render_mode"] = 1
	# cfg["vec"] = make_vec_config(mode, args)
	vecenv = load_env("puffer_drive", cfg)
	return vecenv, cfg

def rollout():
	os.chdir(puffer_root)

	vecenv, cfg = load_pufferdrive_vecenv()

	vecenv.reset(seed=seed) # number_agent = num_agents * batch_size
	reward_sums = []
	env_count = 0

	for _ in range(steps):
		action = vecenv.action_space.sample()
		obs, rewards, terminals, truncations, info = vecenv.step(action)
		imgs = []
		# for i, env in enumerate(vecenv.envs):
			# for j in range(4):
				# img = env.render(view_mode=1, draw_traces=True, env_id=j)
				# vecenv.driver_env.render(view_mode=0, draw_traces=True, env_id=0)
				# print(type(img), img is None)
				# imgs.append(img)
		
		reward_sums.append(float(np.asarray(rewards).sum()))
	for i, env in enumerate(vecenv.envs):
		print(len(env.env_ids))
		env_count += len(env.env_ids)
	print(f"Total agents_per_batch", {vecenv.agents_per_batch})
	print(f"Action Space", {vecenv.action_space.sample().shape})
	print(f"Total environments: {env_count}")
		

	vecenv.close()
	return reward_sums


if __name__ == "__main__":
	rollout()
