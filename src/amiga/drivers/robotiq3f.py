"""Module to control Robotiq's grippers - tested with HAND-E.

Taken from https://github.com/githubuser0xFFFF/py_robotiq_gripper/blob/master/src/robotiq_gripper.py
"""

import socket
import threading
import time
from enum import Enum
from typing import OrderedDict, Tuple, Union



class Robotiq3fGripperOutput:
    def __init__(self):
        #### Byte 0
        # Bit 0
        self.rACT = 0  # 0x0 = deactivate, 0x1 = activate (must stay on until activation is completed)

        # Bits 1,2
        self.rMOD = (
            0  # 0x0 = basic mode, 0x1 = pinch mode, 0x2 = wide mode, 0x3 = scissor mode
        )

        # Bit 3
        self.rGTO = 0  # 0x0 = stop, 0x1 = go to requested position. The only motions performed without the rGTO bit are: activation, the mode change and automatic release routines.

        # Bit 4
        self.rATR = 0  # 0x0 = normal, 0x1 = emergency automatic release in case of a fault. Overrides everything.

        #### Byte 1

        # Bit 2
        self.rICF = 0  # 0x0 = normal control, 0x1 = individual finger control
        # In Individual Control of Fingers Mode each finger receives its own command (position request, speed and force) unless the Gripper is in the
        # Scissor Grasping Mode and the Independent Control of Scissor (rICS) is not activated.

        # Bit 3
        self.rICS = 0  # 0x0 = normal control, 0x1 = individual control of scissor
        # In Individual Control of Scissor, the scissor axis moves independently from the Grasping Mode. When this option is selected, the rMOD bits
        # (Grasping Mode) are ignored as the scissor axis position is defined by the rPRS (Position Request for the Scissor axis) register which takes priority.

        #### Byte 2 is reserved

        #### Byte 3
        # Bits 0 to 7
        self.rPRA = 0  # 0x0 to 0xFF = position request for all fingers, or finger A if individual control mode. 0x0 = open, 0xFF = closed.

        #### Byte 4
        # Bits 0 to 7
        self.rSPA = 129  # 0x0 to 0xFF = speed request for all fingers, or finger A if individual control mode. 0x0 = minimum speed (22mm/s), 0xFF = maximum speed (110 mm/s).

        #### Byte 5
        # Bits 0 to 7
        self.rFRA = 85  # 0x0 to 0xFF = force request for all fingers, or finger A if individual control mode. 0x0 = minimum force (15N), 0xFF = maximum force (60N).

        ######################## Unused registers from advanced control modes, all set to 0 ########################
        #### Bytes 6 to 14
        self.rPRB = 0
        self.rSPB = 129
        self.rFRB = 85
        self.rPRC = 0
        self.rSPC = 129
        self.rFRC = 85
        self.rPRS = 0  # Scissor
        self.rSPS = 129
        self.rFRS = 85

    def __str__(self):
        return f"rACT: {self.rACT}, rMOD: {self.rMOD}, rGTO: {self.rGTO}, rATR: {self.rATR}, rICF: {self.rICF}, rICS: {self.rICS}, rPRA: {self.rPRA}, rSPA: {self.rSPA}, rFRA: {self.rFRA}"


class Robotiq3fGripperInput:
    def __init__(self):
        #### Byte 0
        # Bit 0
        self.gACT = 0  # Echo of the rACT bit
        # 0x00 = gripper is not activated
        # 0x01 = gripper is activated

        # Bits 1, 2
        self.gMOD = 0  # Echo of the rMOD bits
        # 0x00 = basic mode
        # 0x01 = pinch mode
        # 0x02 = wide mode
        # 0x03 = scissor mode

        # Bit 3
        self.gGTO = 0  # Echo of the rGTO bit
        # 0x00 = stopped / performing activation / grasping mode change
        # 0x01 = go to requested position

        # Bits 4, 5
        self.gIMC = 0  # Gripper status
        # 0x00 = stopped
        # 0x01 = activating
        # 0x02 = changing mode
        # 0x03 = activated

        # Bits 6, 7
        self.gSTA = 0  # Motion status;
        # 0x00 = moving to position
        # 0x01 = stopped; 1 or 2 fingers stopped before requested position
        # 0x02 = stopped; all fingers stopped before requested position
        # 0x03 = stopped; reached required position

        #### Byte 1

        # Bits 0, 1
        self.gDTA = 0  # Finger A status
        # 0x00 = finger is moving
        # 0x01 = finger is stopped due to a contact while opening
        # 0x02 = finger is stopped due to a contact while closing
        # 0x03 = finger is at requested position

        # Bits 2, 3
        self.gDTB = 0

        # Bits 4, 5
        self.gDTC = 0

        # Bits 6, 7
        self.gDTS = 0

        #### Byte 2
        # Bits 0 to 3
        self.gFLT = 0  # Fault status
        # 0x00 = no fault
        ######################## Priority faults (fault LED off) #######################
        # 0x05 = action delayed, activation must be completed
        # 0x06 = action delayed, mode change must be completed
        # 0x07 = activation bit must be set prior to action
        #################### Minor faults (fault LED continuous red)####################
        # 0x09 = communication chip not ready (booting?)
        # 0x0A = changing mode fault, interference detected on scissor
        # 0x0B = automatic release in progress
        ############# Major faults (fault LED blinking red), reset required ############
        # 0x0D = activation fault
        # 0x0E = changing mode fault, interference detected on scissor > 20s
        # 0x0F = automatic release completed, reset required

        # Bits 4 to 7 are reserved

        #### Byte 3, all bits
        self.gPRA = 0  # position request echo

        #### Byte 4, all bits
        self.gPOA = 0  # actual position; 0x00 is minimum (open), 0xFF maximum (closed)

        #### Byte 5, all bits
        self.gCUA = 0  # current in figer A

        #### Byte 6 to 14
        self.gPRB = 0
        self.gPOB = 0
        self.gCUB = 0
        self.gPRC = 0
        self.gPOC = 0
        self.gCUC = 0
        self.gPRS = 0
        self.gPOS = 0
        self.gCUS = 0

    def __str__(self):
        return f"gACT: {self.gACT}, gMOD: {self.gMOD}, gGTO: {self.gGTO}, gIMC: {self.gIMC}, gSTA: {self.gSTA}, gDTA: {self.gDTA}, gDTB: {self.gDTB}, gDTC: {self.gDTC}, gDTS: {self.gDTS}, gFLT: {self.gFLT}, gPRA: {self.gPRA}, gPOA: {self.gPOA}, gCUA: {self.gCUA}"


class Robotiq3fGripperModbusTCP:
    """Communicates with the gripper directly, via socket with string commands, leveraging string names for variables."""

    def __init__(self):
        """Constructor."""
        self.client = None

        self._thread = None
        self._stop_event = threading.Event()

        self.grip_input = Robotiq3fGripperInput()  # This is what the gripper sends back
        self.command = Robotiq3fGripperOutput()  # This is what we send to the gripper

        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            raise ImportError(
                "pymodbus is required to use the Robotiq3fGripperModbusTCP wrapper."
            )

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._read_from_gripper()
                self._write_to_gripper()
            except Exception as e:
                print("Error in gripper loop: ", e)
                if not self._reconnect():
                    print("Failed to reconnect to gripper")
                    time.sleep(1.0)
            time.sleep(1.0 / 100.0)

    def _reconnect(self):
        from pymodbus.client import ModbusTcpClient

        self.client = ModbusTcpClient(
            host=self.hostname,
            unit_id=2,
            port=self.port,
            timeout=self.timeout,
            retries=5,
        )
        if not self.client.connect():
            return False

        return True

    def connect(self, hostname: str, port: int, timeout: float = 10.0) -> None:
        """Connects to a gripper at the given address.

        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.hostname = hostname
        self.port = port
        self.timeout = timeout
        from pymodbus.client import ModbusTcpClient

        self.client = ModbusTcpClient(
            host=hostname, port=port, timeout=timeout, retries=5
        )
        if not self.client.connect():
            raise Exception("Failed to connect to gripper at {hostname}:{port}")
        # self.client = ModbusClient(host=hostname, port=port, auto_open=True, timeout=timeout)

        self._read_from_gripper()

        self.command = Robotiq3fGripperOutput()
        self.command.rACT = self.grip_input.gACT
        self.command.rMOD = self.grip_input.gMOD
        self.command.rGTO = self.grip_input.gGTO
        self.command.rPRA = self.grip_input.gPRA
        self.command.rPRB = self.grip_input.gPRB
        self.command.rPRC = self.grip_input.gPRC
        self.command.rPRS = self.grip_input.gPRS

        # print(self.command)

        self.init()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        assert self.client is not None
        self._stop_event.set()

        self.client.close()
        self.client = None

    def _write_to_gripper(self):
        assert self.client is not None

        command = self.command

        # Pack the Action Request register byte
        map = [0] * 16

        map[0] = (
            (command.rACT & 0x1)
            | (command.rMOD << 0x1) & 0x6
            | ((command.rGTO << 0x3) & 0x8)
            | ((command.rATR << 0x4) & 0x10)
        )

        # Pack the Gripper Options register byte
        map[1] = ((command.rICF << 0x2) & 0x4) | ((command.rICS << 0x3) & 0x8)

        # map[2] is empty

        # Requested Position, Speed and Force (Finger A).
        map[3] = command.rPRA
        map[4] = command.rSPA
        map[5] = command.rFRA

        # Finger B
        map[6] = command.rPRB
        map[7] = command.rSPB
        map[8] = command.rFRB

        # Finger C
        map[9] = command.rPRC
        map[10] = command.rSPC
        map[11] = command.rFRC

        # Scissor Mode
        map[12] = command.rPRS
        map[13] = command.rSPS
        map[14] = command.rFRS

        # Pack the registers onto 8 registers using 16bytes
        to_send = [0] * 8
        for i in range(8):
            to_send[i] = (map[i * 2] << 8) | map[i * 2 + 1]

        # Write to the registers
        res = self.client.write_registers(0, to_send)
        if res.isError():
            print(f"Failed to write to output registers")
            raise Exception("Failed to write to output registers: " + str(res))

        # sleep
        time.sleep(0.1)

    def _read_from_gripper(self):
        assert self.client is not None

        # Read the registers
        resp = self.client.read_input_registers(0, 8)
        if resp.isError():
            print(f"Failed to read from input registers: " + str(resp))
            return None

        # Unpack the registers; 1 ModBus register = 2bytes, but 1 gripper register = 1 byte
        res = [0] * 16
        for i, reg_value in enumerate(resp.registers):
            res[i * 2] = (reg_value & 0xFF00) >> 8
            res[i * 2 + 1] = reg_value & 0x00FF

        # Byte 0
        self.grip_input.gACT = res[0] & 0x1
        self.grip_input.gMOD = (res[0] >> 0x1) & 0x3
        self.grip_input.gGTO = (res[0] >> 0x3) & 0x1
        self.grip_input.gIMC = (res[0] >> 0x4) & 0x3
        self.grip_input.gSTA = (res[0] >> 0x6) & 0x3

        # Byte 1
        self.grip_input.gDTA = res[1] & 0x3
        self.grip_input.gDTB = (res[1] >> 0x2) & 0x3
        self.grip_input.gDTC = (res[1] >> 0x4) & 0x3
        self.grip_input.gDTS = (res[1] >> 0x6) & 0x3

        # Byte 2
        self.grip_input.gFLT = res[2] & 0xF

        # Byte 3
        self.grip_input.gPRA = res[3]

        # Byte 4
        self.grip_input.gPOA = res[4]

        # Byte 5
        self.grip_input.gCUA = res[5]

        # Byte 6 to 14
        self.grip_input.gPRB = res[6]
        self.grip_input.gPOB = res[7]
        self.grip_input.gCUB = res[8]
        self.grip_input.gPRC = res[9]
        self.grip_input.gPOC = res[10]
        self.grip_input.gCUC = res[11]
        self.grip_input.gPRS = res[12]
        self.grip_input.gPOS = res[13]
        self.grip_input.gCUS = res[14]

        return self.grip_input

    def _is_halted(self):
        # self._read_from_gripper()
        return self.grip_input.gGTO == 0

    def _is_ready(self):
        # self._read_from_gripper()
        return self.grip_input.gIMC == 3

    def _is_moving(self):
        # self._read_from_gripper()
        # print(self.grip_input.gSTA)
        return self.grip_input.gSTA == 0

    def _is_emergency_release_complete(self):
        # self._read_from_gripper()
        return self.grip_input.gFLT == 0xF

    def _is_initialised(self):
        # self._read_from_gripper()
        return self.grip_input.gACT == 1

    def init(self) -> bool:
        """Initializes the gripper."""
        self.command.rACT = 1

        # print(self.command)
        # self._write_to_gripper()

        # self._read_from_gripper()
        # # print(self.grip_input)

        self.command.rACT = 1
        # self._write_to_gripper()

        # Wait for initialisation to complete
        c = 0
        print("waiting for gripper initialisation")
        while not self._is_ready():
            time.sleep(0.1)
            c += 1
            if c > 500:
                print("Failed to initialise gripper")
                return False
        print("gripper initialised")

        # Set the gripper to basic mode
        self.command.rMOD = 0
        self.command.rGTO = 1
        self.command.rICF = 0
        self.command.rICS = 0
        # self._write_to_gripper()

        # wait for halt
        c = 0
        print("waiting for gripper to be ready to move")
        while self._is_halted():
            time.sleep(0.1)
            c += 1
            if c > 500:
                print("Failed to set gripper to basic mode")
                return False

        print("gripper ready")
        # Set the speed and force
        self.command.rSPA = 255
        self.command.rFRA = 255
        # self._write_to_gripper()

        return True

    def close(self) -> bool:
        if not self._is_initialised():
            print("Gripper not initialised")
            self.init()
            # return False

        self.command.rPRA = 255
        self.command.rMOD = 0
        self.command.rGTO = 1

        return True

    def open(self) -> bool:
        if not self._is_initialised():
            print("Gripper not initialised")
            self.init()
            # return False

        self.command.rPRA = 0
        self.command.rMOD = 0
        self.command.rGTO = 1

        return True

    def halt(self) -> bool:
        if not self._is_initialised():
            print("Gripper not initialised")
            return False

        self.command.rGTO = 0
        # self._write_to_gripper(self.command)

        c = 0
        while not self._is_halted():
            time.sleep(0.1)
            c += 1
            if c > 500:
                print("Failed to halt gripper")
                return False
        return True

    def emergency_release(self) -> bool:
        if not self.halt():
            return False

        self.command.rATR = 1
        # self._write_to_gripper()

        c = 0
        while not self._is_emergency_release_complete():
            time.sleep(0.1)
            c += 1
            if c > 500:
                print("Failed to complete emergency release")
                return False

        self.command.rATR = 0

        return True

    def shutdown(self) -> bool:
        if not self.halt():
            return False

        self.command.rACT = 0
        # self._write_to_gripper()

        c = 0
        while self._is_initialised():
            time.sleep(0.1)
            c += 1
            if c > 500:
                print("Failed to shutdown gripper")
                return False

        return True

    def reset(self) -> bool:
        if not self.shutdown():
            return False
        if not self.init():
            return False
        return True

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position, with the specified speed and force.

        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """
        # Clip commands to 0, 255
        position = int(max(min(position, 255), 0))
        speed = int(max(min(speed, 255), 0))
        force = int(max(min(force, 255), 0))

        if not self._is_initialised():
            print("Gripper not initialised")
            return False, 0

        # print("Gripper to : ", position, speed, force)

        self.command.rPRA = position
        self.command.rSPA = speed
        self.command.rFRA = force
        self.command.rGTO = 1
        # self._write_to_gripper()

        # c = 0
        # while not self._is_moving():
        #     time.sleep(0.1)
        #     c +=1
        #     if c > 500:
        #         print("Failed to start moving")
        #         return False, 0

        return True, position

    def get_current_position(self) -> int:
        """Returns the current position of the gripper."""
        # self._read_from_gripper()
        return self.grip_input.gPOA
    
    def has_object(self) -> bool:
        """Returns True if the gripper has an object."""
        # print("gDTA: ", self.grip_input.gDTA)
        # print("gDTB: ", self.grip_input.gDTB)
        # print("gDTC: ", self.grip_input.gDTC)
        # print("gGTO: ", self.grip_input.gGTO)
        return (
            self.grip_input.gGTO == 1 and self.grip_input.gDTA == 2,
            self.grip_input.gGTO == 1 and self.grip_input.gDTB == 2,
            self.grip_input.gGTO == 1 and self.grip_input.gDTC == 2
        )


def main():
    # test open and closing the gripper
    gripper = Robotiq3fGripperModbusTCP()
    gripper.connect(hostname="192.168.1.10", port=63352)
    # gripper.activate()
    print(gripper.get_current_position())
    gripper.move(20, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(65, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(20, 255, 1)
    gripper.disconnect()


if __name__ == "__main__":
    main()
