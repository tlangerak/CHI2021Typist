import random
import time

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

ip = "192.168.0.102"
port = 12000
client = SimpleUDPClient(ip, port)  # Create client
w = 1
h = 1


def print_handler(address, *args):
    print(f"{address}: {args}")


def init_handler(address, *args):
    global w, h
    print(f"INIT {address}: {args}")
    w = args[0]
    h = args[1]


def default_handler(address, *args):
    client.send_message("/new_target", [random.randint(0, w),
                                        random.randint(0, h),
                                        random.randint(10, 50),
                                        random.randint(10, 50)])  # Send message with int, float and string
    print(f"DEFAULT {address}: {args}")


if __name__ == "__main__":
    dispatcher = Dispatcher()
    dispatcher.map("/something/*", print_handler)
    dispatcher.map("/init", init_handler)

    dispatcher.set_default_handler(default_handler)

    ip = "192.168.0.100"
    port = 5005

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    server.serve_forever()  # Blocks forever
