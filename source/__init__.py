# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# NOTE: this file did not exist in the uploaded source and was added while packaging
# the repo, so that the configs are actually reachable as Gym environments. It follows
# the standard IsaacLab manager-based-task registration pattern. Double check the
# rsl_rl_cfg_entry_point path against whatever agent config you're actually using
# (none was uploaded, so it's left pointing at a conventional agents/rsl_rl_ppo_cfg.py
# that you'll need to add).

import gymnasium as gym

from . import flat_env_cfg, rough_env_cfg

gym.register(
    id="Isaac-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.G1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.G1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)
