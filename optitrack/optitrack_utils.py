from pyquaternion import Quaternion
from NatNetClient import NatNetClient
import numpy as np
import time


class RigidBodyDataModel:
    def __init__(self, id):
        self.rotation = None
        self.position = None
        self.ID = id
        self.is_active = False

    def update(self, pos, rot):
        self.is_active = True
        self.rotation = rot
        self.position = pos


class OptitrackClient:
    def __init__(self):
        self.streamingClient = None
        max_num_rigid_bodies = 10

        self.rigidbodies = []
        for i in range(1, max_num_rigid_bodies):
            self.rigidbodies.append(RigidBodyDataModel(i))

    def start(self):
        # This will create a new NatNet client
        self.streamingClient = NatNetClient(server="127.0.0.1",
                                            rigidBodyListener=self.receiveRigidBodyFrame,
                                            # newFrameListener = receiveNewFrame,
                                            multicast="239.255.42.99",
                                            dataPort=1511,
                                            commandPort=1510,
                                            verbose=False)

        try:
            self.streamingClient.run()
            # while True:            #     time.sleep(1)
            #     print (self.get_current_camera_pos_rot(False))
        except (KeyboardInterrupt, SystemExit):
            print("Shutting down natnet interfaces...")
            self.streamingClient.stop()
        except OSError:
            print("Natnet connection error")
            self.streamingClient.stop()
            exit(-1)

    def stop(self):
        if self.streamingClient is not None:
            self.streamingClient.stop()

    def get_current_pos_rot_for_rigidbody(self, id):
        return self.rigidbodies[id].position, self.rigidbodies[id].rotation

    def get_current_pos_rot_for_rigidbody_fake(self, id):
        return np.array([.25 * id, 0, .25 * id]), Quaternion(1, 0, 0, 0)

    # This is a callback function that gets connected to the NatNet client and called once per mocap frame.
    def receiveNewFrame(self, frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                        labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged):
        print("Received frame", frameNumber)

    def receiveRigidBodyFrame(self, id, pos_raw, rot_raw):
        # print( "Received frame for rigid body", id )

        # Note that Natnet gives quaternions as x,y,z,w; pyquaternion need w,x,y,z. This is taken care of here.
        self.rigidbodies[id].update(np.array(pos_raw), Quaternion(rot_raw[3], rot_raw[0], rot_raw[1], rot_raw[2]))

if __name__ == '__main__':
    sampleClient = OptitrackClient()
    sampleClient.start()
    try:
        while True:
            time.sleep(1)
            # print (sampleClient.get_current_board_positions(False))
            # print (sampleClient.get_current_board_positions_fake(False))
            print(sampleClient.get_current_pos_rot_for_rigidbody(False))
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down natnet interfaces...")
        sampleClient.stop()
    except OSError:
        print("Natnet connection error")
        sampleClient.stop()
        exit(-1)