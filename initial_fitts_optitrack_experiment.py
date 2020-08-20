import random
import time
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pyquaternion import Quaternion
from optitrack.NatNetClient import NatNetClient
from optitrack.optitrack_utils import RigidBodyDataModel, OptitrackClient
import json

participant_id = "p0"

ip = "192.168.0.109"
port = 12000
client = SimpleUDPClient(ip, port)  # Create client

w_min = 10
h_min = 10
w_max = 100
h_max = 100

x_target = 0
y_target = 0
w_target = 0
h_target = 0
x_old = 0
y_old = 0
w_old = 0
h_old = 0
min_size = 10
max_size = 50

start_time = 0
end_time = 0
start_coordinate = [0, 0]
end_coordinate = [0, 0]

filename = "fittslaw_simple_{}.json".format(participant_id)
repitition_counter = 0


def print_handler(address, *args):
    print(f"{address}: {args}")
    return


def init_handler(address, *args):
    global w_max, h_max
    print(f"INIT {address}: {args}")
    w_max = args[0]
    h_max = args[1]
    return


def calculate_hit(touch, target):
    if touch[0] <= target[0] + target[2] / 2 and touch[0] >= target[0] - target[2] / 2:
        if touch[1] <= target[1] + target[3] / 2 and touch[1] >= target[1] - target[3] / 2:
            return True
    return False


def calculate_distance(touch, target):
    return ((touch[0] - target[0]) ** 2 + (touch[1] - target[1]) ** 2) ** 0.5


def hit_handler(address, *args):
    global x_target, y_target, w_target, h_target, x_old, y_old, w_old, h_old
    global end_time, end_coordinate, repitition_counter
    end_coordinate = [args[0], args[1]]
    end_time = args[2]

    hit = calculate_hit(end_coordinate, [x_target, y_target, w_target, h_target])
    distance = calculate_distance(end_coordinate, [x_target, y_target])
    update_data = {
        "repetition_{}".format(repitition_counter): {
            "movement_time": end_time - start_time,
            "target_x": x_target,
            "target_y": y_target,
            "target_w": w_target,
            "target_h": h_target,
            "start_x": start_coordinate[0],
            "start_y": start_coordinate[1],
            "end_x": end_coordinate[0],
            "end_y": end_coordinate[1],
            "hit": hit,
            "distance_to_center": distance,
        }
    }
    json_write(filename, update_data)
    x_old = x_target
    y_old = y_target

    if repitition_counter %2 == 0:
        while x_old == x_target and y_old == y_target:
            x = [300, 960, 1620]
            if x == 960:
                y = [300, 900]
            else:
                y = [300, 600, 900]
            s = [50, 150, 200]
            w_ind = random.randint(0, len(s)-1)
            x_ind = random.randint(0, len(x)-1)
            y_ind = random.randint(0, len(y)-1)
            w_target = s[w_ind]
            h_target = s[w_ind]
            y_target = y[y_ind]
            x_target = x[x_ind]
    else:
        x_target = 960
        y_target = 600
        w_target = 25
        h_target = 25

    print(x_target, y_target, w_target)
    client.send_message("/new_target", [x_target,
                                        y_target,
                                        w_target,
                                        h_target])  # Send message with int, float and string
    print(f"HIT {address}: {args}")
    repitition_counter += 1
    return


def lift_handler(address, *args):
    global start_time, start_coordinate
    start_coordinate = [args[0], args[1]]
    start_time = args[2]
    print(start_time, start_coordinate)
    pass


def json_write(filen: str, data: dict):
    with open(filen, 'r') as json_file:
        z = json.load(json_file)
        z.update(data)
    with open(filen, 'w') as json_file:
        json.dump(z, json_file, indent=4, sort_keys=True)


def json_init(filen: str):
    init_data = {"meta": {
        "date_time": int(time.time()),
        "participant": participant_id,
        "tester": "Thomas",
        "iteration": 1,
        "sample_time": 0.001
    }
    }

    with open(filen, 'w') as outfile:
        json.dump(init_data, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    print(filename)
    json_init(filename)
    dispatcher = Dispatcher()
    dispatcher.map("/hit", hit_handler)
    dispatcher.map("/release", lift_handler)
    dispatcher.map("/init", init_handler)

    dispatcher.set_default_handler(print_handler)

    ip = "192.168.0.108"
    port = 5005

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    server.serve_forever()  # Blocks forever
