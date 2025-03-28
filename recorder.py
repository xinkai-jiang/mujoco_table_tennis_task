import mujoco
import os
import numpy as np
import pickle
from typing import List, Dict

from simpub.xr_device.meta_quest3 import MetaQuest3


class EnvRecorder:

    def __init__(self, mj_model, mj_data, object_names: List[str]):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.index = 0
        self.recording_dir = "./recording_data"
        os.makedirs(self.recording_dir, exist_ok=True)
        self.object_names = object_names
        self.object_ids = []
        for object_name in object_names:
            object_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            assert object_id >= 0, f"Object {object_name} not found in model"
            self.object_ids.append(object_id)
        self.data = []

    def start_recording(self):
        # create a directory to store the recorded data
        self.data = []

    def record_obs(self) -> Dict:
        step_obs = {}
        for obj_name, obj_id in zip(self.object_names, self.object_ids):
            step_obs[obj_name] = {
                "xpos": self.mj_data.xpos[obj_id].copy(),
                "xquat": self.mj_data.xquat[obj_id].copy(),
            }
        return step_obs

    def record_step(self, xr_device: MetaQuest3):
        input_data = xr_device.get_input_data()
        if input_data is None:
            input_data = {}
        step_data = {
            "obs": self.record_obs(),
            "time": self.mj_data.time,
            "input_data": input_data,
        }
        self.data.append(step_data)

    def save_data(self):
        file_name = f"recording_{self.index}.pickle"
        print(f"Saving data to {os.path.join(self.recording_dir, file_name)}")
        # np.savez(os.path.join(self.recording_dir, file_name), self.data)
        with open(os.path.join(self.recording_dir, file_name), "wb") as f:
            pickle.dump(self.data, f)
        self.index += 1
        self.data = []
