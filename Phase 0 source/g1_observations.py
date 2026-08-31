import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

def imu_ang_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body-frame angular velocity from IMU (gyro)."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.ang_vel_b


def imu_projected_gravity(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gravity vector in body frame, from IMU-derived orientation."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.projected_gravity_b

def _ensure_imu_bias_buffer(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "_imu_acc_bias"):
        env._imu_acc_bias = torch.zeros(env.num_envs, 3, device=env.device)


def imu_lin_acc_biased(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    bias_std: float = 0.15,
) -> torch.Tensor:
    """Linear acceleration with persistent bias (drawn once per episode)."""
    imu = env.scene[sensor_cfg.name]
    _ensure_imu_bias_buffer(env)
    return imu.data.lin_acc_b + env._imu_acc_bias


def reset_imu_bias(env: ManagerBasedRLEnv, env_ids: torch.Tensor, bias_std: float = 0.15) -> None:
    """EventTerm (mode='reset'): redraw accelerometer bias for reset envs."""
    _ensure_imu_bias_buffer(env)
    env._imu_acc_bias[env_ids] = torch.randn(len(env_ids), 3, device=env.device) * bias_std


def foot_contact_booleans(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
    flip_prob: float = 0.05,
) -> torch.Tensor:
    """Binary contact flags for feet, with label noise so this isn't a
    perfect oracle signal. Returns (num_envs, num_feet)."""
    sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    contact = (forces > force_threshold).float()
    flip_mask = torch.rand_like(contact) < flip_prob
    return torch.where(flip_mask, 1.0 - contact, contact)

_EKF_Q = 0.5          # process noise (acceleration integration uncertainty)
_EKF_R = 0.05         # measurement noise (leg odometry uncertainty)
_EKF_P0 = 1.0         # initial covariance
_EKF_V_CLAMP = 10.0   # max |velocity| before clamping
_EKF_P_CLAMP = 5.0    # max covariance before clamping
_GRAVITY_WORLD = torch.tensor([0.0, 0.0, -9.81])  # world-frame gravity vector

def _ensure_ekf_buffers(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "_ekf_v"):
        env._ekf_v = torch.zeros(env.num_envs, 3, device=env.device)
        env._ekf_P = torch.full((env.num_envs, 3), _EKF_P0, device=env.device)
        env._ekf_last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)


def reset_ekf_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """EventTerm (mode='reset'): zero EKF state for reset envs."""
    _ensure_ekf_buffers(env)
    env._ekf_v[env_ids] = 0.0
    env._ekf_P[env_ids] = _EKF_P0
    env._ekf_last_step[env_ids] = -1


def _leg_odometry_velocity(
    env: ManagerBasedRLEnv,
    robot,
    imu,
    asset_cfg: SceneEntityCfg,
    contact_forces,
    force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_ids = asset_cfg.body_ids
    joint_ids = asset_cfg.joint_ids
    num_feet = len(body_ids)

    omega_b = imu.data.ang_vel_b
    quat_wb = imu.data.quat_w

    # Contact flags: sensor may track all bodies, index with foot body_ids
    if contact_forces.data.net_forces_w.shape[1] == num_feet:
        forces = torch.norm(contact_forces.data.net_forces_w, dim=-1)
    else:
        forces = torch.norm(contact_forces.data.net_forces_w[:, body_ids], dim=-1)
    in_stance = (forces > force_threshold).float()

    # Jacobians: try IsaacLab data accessors first, fall back to PhysX view
    if hasattr(robot.data, "body_jacobians_w"):
        jacobians = robot.data.body_jacobians_w
        jac_joint_ids = joint_ids
        jacobian_frame = "world"
    elif hasattr(robot.data, "body_jacobians"):
        jacobians = robot.data.body_jacobians
        if jacobians.shape[-1] == len(joint_ids):
            jac_joint_ids = joint_ids
            jacobian_frame = "world"
        else:
            jac_joint_ids = [j + 6 for j in joint_ids]
            jacobian_frame = "body"
    else:
        jacobians = robot.root_physx_view.get_jacobians()
        if jacobians.shape[-1] == robot.num_joints:
            jac_joint_ids = joint_ids
            jacobian_frame = "world"
        else:
            jac_joint_ids = [j + 6 for j in joint_ids]
            jacobian_frame = "body"

    joint_vel = robot.data.joint_vel

    est_vels = torch.zeros(env.num_envs, num_feet, 3, device=env.device)

    if jacobian_frame == "world":
        omega_w = quat_apply(quat_wb, omega_b)
        foot_pos_w = robot.data.body_pos_w[:, body_ids]
        base_pos_w = robot.data.root_pos_w.unsqueeze(1)
        r_foot = foot_pos_w - base_pos_w

        for i, body_id in enumerate(body_ids):
            J_full = jacobians[:, body_id, :, :]
            J_lin = J_full[:, :3, :][:, :, jac_joint_ids]
            foot_vel_from_joints = torch.einsum("bij,bj->bi", J_lin, joint_vel[:, joint_ids])
            omega_cross_r = torch.cross(omega_w, r_foot[:, i], dim=-1)
            v_base_est = -omega_cross_r - foot_vel_from_joints
            est_vels[:, i] = v_base_est

    else:
        foot_pos_w = robot.data.body_pos_w[:, body_ids]
        base_pos_w = robot.data.root_pos_w.unsqueeze(1)
        r_foot_world = foot_pos_w - base_pos_w
        r_foot_body = quat_apply_inverse(
            quat_wb.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4),
            r_foot_world.reshape(-1, 3),
        ).reshape(-1, num_feet, 3)

        for i, body_id in enumerate(body_ids):
            J_full = jacobians[:, body_id, :, :]
            J_lin = J_full[:, :3, :][:, :, jac_joint_ids]
            foot_vel_from_joints_b = torch.einsum("bij,bj->bi", J_lin, joint_vel[:, joint_ids])
            omega_cross_r_b = torch.cross(omega_b, r_foot_body[:, i], dim=-1)
            v_base_est_b = -omega_cross_r_b - foot_vel_from_joints_b
            est_vels[:, i] = quat_apply(quat_wb, v_base_est_b)

    weights = in_stance.unsqueeze(-1)
    weight_sum = weights.sum(dim=1).clamp(min=1e-6)
    v_leg_odom = (est_vels * weights).sum(dim=1) / weight_sum
    has_stance = in_stance.sum(dim=1) > 0
    return v_leg_odom, has_stance


def estimated_base_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    imu_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """EKF that fuses IMU integration with leg-odometry zero-velocity updates."""
    _ensure_ekf_buffers(env)
    current_step = getattr(env, "common_step_counter", 0)
    already_computed = env._ekf_last_step == current_step
    if already_computed.all():
        return env._ekf_v
    active = ~already_computed
    # -------------------------------------------------------

    robot = env.scene[asset_cfg.name]
    imu = env.scene[imu_sensor_cfg.name]
    contact_forces = env.scene[contact_sensor_cfg.name]
    dt = getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation)

    quat_wb = imu.data.quat_w
    acc_world = quat_apply(quat_wb, imu.data.lin_acc_b) + _GRAVITY_WORLD.to(env.device)
    
    v_pred = env._ekf_v.clone()
    P_pred = env._ekf_P.clone()
    v_pred[active] = v_pred[active] + acc_world[active] * dt
    P_pred[active] = P_pred[active] + _EKF_Q * dt

    v_meas, has_stance = _leg_odometry_velocity(
        env, robot, imu, asset_cfg, contact_forces, force_threshold
    )

    K = P_pred / (P_pred + _EKF_R)
    innovation = v_meas - v_pred
    v_updated = v_pred + K * innovation
    P_updated = (1.0 - K) * P_pred

    stance_mask = has_stance.unsqueeze(-1)
    
    new_v = torch.where(stance_mask, v_updated, v_pred)
    new_P = torch.where(stance_mask, P_updated, P_pred)
    
    env._ekf_v = torch.where(active.unsqueeze(-1), new_v, env._ekf_v)
    env._ekf_P = torch.where(active.unsqueeze(-1), new_P, env._ekf_P)
    env._ekf_last_step[active] = current_step

    env._ekf_v = torch.clamp(env._ekf_v, -_EKF_V_CLAMP, _EKF_V_CLAMP)
    env._ekf_P = torch.clamp(env._ekf_P, 1e-6, _EKF_P_CLAMP)

    return env._ekf_v
    
def feet_phase_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Reward alternating gait: +1 when exactly one foot is in contact.
    Penalizes double stance and double flight. Returns (num_envs,)."""
    sensor = env.scene[sensor_cfg.name]
    if sensor.data.net_forces_w.shape[1] == 2:
        forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    else:
        forces = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    
    contact = (forces > force_threshold).float()   # (N, 2)
    num_in_contact = contact.sum(dim=1)            # (N,)
    
    # Peak reward when exactly 1 foot is in contact (alternating)
    return 1.0 - (num_in_contact - 1.0).abs()


def feet_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Reward lifting feet during swing phase. Returns (num_envs,)."""
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    body_ids = asset_cfg.body_ids
    
    # Foot height in world frame (Z coordinate)
    foot_heights = robot.data.body_pos_w[:, body_ids, 2]  # (N, num_feet)
    
    # Contact flags
    if sensor.data.net_forces_w.shape[1] == len(body_ids):
        forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    else:
        forces = torch.norm(sensor.data.net_forces_w[:, body_ids], dim=-1)
    in_stance = (forces > force_threshold).float()  # (N, num_feet)
    
    # Reward height only during swing (not in contact)
    # Linear reward: 0 at 2cm, max at target_height
    swing_mask = 1.0 - in_stance
    height_above_min = torch.clamp(foot_heights - 0.02, min=0.0, max=target_height)
    
    return (height_above_min * swing_mask).sum(dim=1)


def joint_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize left/right leg joint position asymmetry. Returns (num_envs,)."""
    robot = env.scene[left_asset_cfg.name]
    left_ids = left_asset_cfg.joint_ids
    right_ids = right_asset_cfg.joint_ids
    
    if len(left_ids) != len(right_ids):
        return torch.zeros(env.num_envs, device=env.device)
    
    joint_pos = robot.data.joint_pos  # (N, num_dof)
    diff = joint_pos[:, left_ids] - joint_pos[:, right_ids]  # (N, num_leg_joints)
    return torch.mean(diff.abs(), dim=1)
