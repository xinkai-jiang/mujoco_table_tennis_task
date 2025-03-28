from mujoco import mj_name2id, mjtObj  # type: ignore
import numpy as np
from scipy.spatial.transform import Rotation

from simpub.xr_device.meta_quest3 import MetaQuest3


def check_episode_and_rest(mj_model, mj_data) -> bool:
    target_ball_id = mj_name2id(mj_model, mjtObj.mjOBJ_BODY, "target_ball")
    object_position = mj_data.xpos[target_ball_id]
    if abs(object_position[0]) > 2 or abs(object_position[1]) > 2:
        reset_ball(mj_model, mj_data)
        return True
    return False


def reset_ball(mj_model, mj_data):
    print("Resetting ball")
    target_ball_id = mj_name2id(mj_model, mjtObj.mjOBJ_BODY, "target_ball")
    body_jnt_addr = mj_model.body_jntadr[target_ball_id]
    assert body_jnt_addr >= 0, "Body joint address not found"
    qposadr = mj_model.jnt_qposadr[body_jnt_addr]
    mj_data.qpos[qposadr: qposadr + 3] = np.array([0, 0, 1.5])
    mj_data.qvel[qposadr: qposadr + 3] = np.array([2.0, 0, 0])


class MujocoFreeObjectTracker:

    def __init__(self, mj_model, mj_data, object_name, xr_device: MetaQuest3):
        self.xr_device = xr_device
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.object_id = mj_name2id(mj_model, mjtObj.mjOBJ_BODY, object_name)
        assert self.object_id >= 0, f"Object {object_name} not found in model"
        self.body_jnt_addr = mj_model.body_jntadr[self.object_id]
        assert self.body_jnt_addr >= 0, f"Joint not found for {object_name}"
        self.dofadr = mj_model.jnt_dofadr[self.body_jnt_addr]
        # the controller gains
        self.pos_gain = 200.0  # Position tracking gain
        self.rot_gain = 0.5  # Rotation tracking gain
        self.max_vel = 100.0  # Maximum linear velocity
        self.max_angvel = 40.0  # Maximum angular velocity

    def _update(self, target_pos, target_quat):
        # pos error
        pos_error = target_pos - self.mj_data.xpos[self.object_id]
        desired_vel = self.pos_gain * pos_error
        vel_norm = np.linalg.norm(desired_vel)
        if vel_norm > self.max_vel:
            desired_vel = desired_vel * (self.max_vel / vel_norm)
        self.mj_data.qvel[self.dofadr: self.dofadr + 3] = desired_vel
        # quat error
        target_rot = Rotation.from_quat(target_quat)
        current_quat = self.mj_data.xquat[self.object_id].copy()
        current_rot = Rotation.from_quat(current_quat[np.array([1, 2, 3, 0])])
        rot_error = current_rot.inv() * target_rot
        desired_angular_vel = self.rot_gain * rot_error.as_rotvec(degrees=True)
        ang_vel_norm = np.linalg.norm(desired_angular_vel)
        if ang_vel_norm > self.max_angvel:
            desired_angular_vel = desired_angular_vel * (self.max_angvel / ang_vel_norm)
        self.mj_data.qvel[self.dofadr + 3: self.dofadr + 6] = desired_angular_vel

    def update(self, hand: str = "right"):
        player_input = self.xr_device.get_input_data()
        if player_input is None:
            return
        target_pos = np.array(player_input[hand]["pos"])
        target_quat = np.array(player_input[hand]["rot"])
        self._update(target_pos, target_quat)
