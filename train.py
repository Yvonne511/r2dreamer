import atexit
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import gymnasium as gym
import hydra
import numpy as np
import torch

import tools
from buffer import Buffer, OfflineDatasetBuffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    print("Create envs.")
    if config.offline:
        dataset = hydra.utils.instantiate(config.env.dataset)
        replay_buffer = OfflineDatasetBuffer(config, dataset)
        train_envs, eval_envs = None, None
        sample = dataset[0]
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, shape=tuple(sample["image"].shape[1:]), dtype=np.uint8),
                "reward": gym.spaces.Box(-np.inf, np.inf, shape=tuple(sample["reward"].shape[1:]), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, shape=tuple(sample["is_first"].shape[1:]), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, shape=tuple(sample["is_last"].shape[1:]), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, shape=tuple(sample["is_terminal"].shape[1:]), dtype=bool),
                **{
                    key: gym.spaces.Box(-np.inf, np.inf, shape=tuple(sample[key].shape[1:]), dtype=np.float32)
                    for key in sorted(sample)
                    if key.startswith("reward_")
                },
            }
        )
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=tuple(sample["action"].shape[1:]), dtype=np.float32)
    else:
        replay_buffer = Buffer(config.buffer)
        train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_envs,
        eval_envs,
        offline=bool(config.offline),
    )
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
