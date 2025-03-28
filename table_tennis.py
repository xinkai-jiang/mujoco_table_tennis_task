import mujoco
import time
import os
import argparse
import mujoco.viewer

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3

from utils import check_episode_and_rest, reset_ball, MujocoFreeObjectTracker
from recorder import EnvRecorder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.134")
    args = parser.parse_args()
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets/table_tennis_env.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    last_time = time.time()
    publisher = MujocoPublisher(model, data, args.host)
    player1 = MetaQuest3("IRLMQ3-1")
    tracker = MujocoFreeObjectTracker(model, data, "bat1", player1)
    count = 0
    recorder = EnvRecorder(model, data, ["target_ball", "bat1"])
    reset_ball(model, data)
    recorder.start_recording()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            try:
                mujoco.mj_step(model, data)
                interval = time.time() - last_time
                if interval < 0.002:
                    time.sleep(0.002 - interval)
                last_time = time.time()
                recorder.record_step(player1)
                tracker.update()
                # update_bat(model, data, player1, "bat1")
                # if count % 10 == 0:
                if check_episode_and_rest(model, data):
                    recorder.save_data()
                    recorder.start_recording()
                # count += 1
                viewer.sync()
            except KeyboardInterrupt:
                break
    publisher.shutdown()
