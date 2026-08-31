# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ImuCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import EventTermCfg as EventTerm
from . import g1_observations as g1_obs

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )

    # ---- GAIT SYMMETRY REWARDS (NEW) ----
    feet_phase = RewTerm(
        func=g1_obs.feet_phase_reward,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 5.0,
        },
    )
    feet_height = RewTerm(
        func=g1_obs.feet_height_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "target_height": 0.05,
            "force_threshold": 5.0,
        },
    )
    joint_symmetry = RewTerm(
        func=g1_obs.joint_symmetry_penalty,
        weight=-0.15,
        params={
            "left_asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_yaw_joint",
                    "left_hip_roll_joint",
                    "left_hip_pitch_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
            ),
            "right_asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_hip_yaw_joint",
                    "right_hip_roll_joint",
                    "right_hip_pitch_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
            ),
        },
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_knee_joint", ".*_hip_pitch_joint"],
            )
        },
    )

@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ==================== CUSTOM STATE ESTIMATION ====================

        # 1. Add IMU sensor to the scene
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            debug_vis=False,
        )

        # 2. Temporal history for the policy (set on ObsGroup, NOT on ObsTerm)
        self.observations.policy.history_length = 3

        # 3. Remove privileged "cheat" observation from physics engine
        self.observations.policy.base_lin_vel = None

        # 4. Replace with IMU-derived and estimated observations
        self.observations.policy.base_ang_vel = ObsTerm(
            func=g1_obs.imu_ang_vel,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        self.observations.policy.projected_gravity = ObsTerm(
            func=g1_obs.imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        self.observations.policy.imu_lin_acc = ObsTerm(
            func=g1_obs.imu_lin_acc_biased,
            params={"sensor_cfg": SceneEntityCfg("imu"), "bias_std": 0.15},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        self.observations.policy.estimated_base_lin_vel = ObsTerm(
            func=g1_obs.estimated_base_lin_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=".*_ankle_roll_link",
                    joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
                ),
                "imu_sensor_cfg": SceneEntityCfg("imu"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "force_threshold": 5.0,
            },
        )
        self.observations.policy.foot_contacts = ObsTerm(
            func=g1_obs.foot_contact_booleans,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "force_threshold": 5.0,
                "flip_prob": 0.05,
            },
        )

        # 5. Reset events for episodic state (EKF + IMU bias)
        self.events.reset_imu_bias = EventTerm(
            func=g1_obs.reset_imu_bias,
            mode="reset",
            params={"bias_std": 0.15},
        )
        self.events.reset_ekf_state = EventTerm(
            func=g1_obs.reset_ekf_state,
            mode="reset",
        )

        # ================================================================

        # Scene
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # ---- GAIT FIXES: boost air time, lower threshold ----
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.2

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
