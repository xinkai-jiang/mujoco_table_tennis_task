import mujoco
from mujoco import mj_name2id, mjtObj
import mujoco.viewer
import numpy as np
import pickle
import os
import time

class Replayer:

    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, recording_dir: str):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.recording_dir = recording_dir
        self.file_list = os.listdir(recording_dir)
        self.file_index = 0
        self.index = 0
        self.data = []

    def load_data(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded data from {file_path}")
        # print(self.data)

    def load_next_data(self) -> bool:
        if self.file_index >= len(self.file_list):
            return True
        file_path = os.path.join(self.recording_dir, f"record_{self.file_index}.pickle")
        self.load_data(file_path)
        self.file_index += 1
        self.index = 0
        return False

    def replay_next_step(self) -> bool:
        if self.index >= len(self.data):
            return True
        # print(f"Replaying step {self.index}")
        obs = self.data[self.index]["obs"]
        for obj_name, obj_data in obs.items():
            obj_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            body_jnt_addr = self.mj_model.body_jntadr[obj_id]
            qposadr = self.mj_model.jnt_qposadr[body_jnt_addr]
            self.mj_data.qpos[qposadr: qposadr + 3] = np.array(obj_data["xpos"])
            self.mj_data.qvel[qposadr: qposadr + 3] = np.array([0, 0, 0])
            self.mj_data.qpos[qposadr + 3: qposadr + 7] = obj_data["xquat"]
            self.mj_data.qpos[qposadr + 3: qposadr + 7] = np.array([0, 0, 0, 1])
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.index += 1
        return False



if __name__ == '__main__':
    data_dir = "./recording_data"
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets/table_tennis_env.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    replayer = Replayer(model, data, data_dir)
    replayer.load_next_data()
    replay_time_step = 0.002
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_time = time.time()
        while True:
            try:
                # mujoco.mj_step(model, data)
                if replayer.replay_next_step():
                    if input("Save this episode? (y/n): ") == "n":
                        os.remove(os.path.join(data_dir, f"record_{replayer.file_index - 1}.pickle"))
                    if replayer.load_next_data():
                        break
                interval = time.time() - last_time
                if interval < replay_time_step:
                    time.sleep(replay_time_step - interval)
                last_time = time.time()
                viewer.sync()
            except KeyboardInterrupt:
                break