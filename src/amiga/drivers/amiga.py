from typing import Dict, Tuple
import time

import numpy as np
from omegaconf import OmegaConf

from amiga.drivers.zmq import ZMQBackendObject


class AMIGA(ZMQBackendObject):
    """A class representing a AMIGA."""

    def __init__(self, cfg: OmegaConf):
        if not self._check_required_libraries() or not self._check_cfg(cfg):
             # This class might be instantiated by a client, just to know its methods; so no error here
            return
        
        self.cfg = cfg
        import rtde_control
        import rtde_receive

        connected = False
        while not connected:
            try:
                self.robot = rtde_control.RTDEControlInterface(cfg.robot_ip, rt_priority=97)
                self.r_inter = rtde_receive.RTDEReceiveInterface(cfg.robot_ip)
                connected = True
            except RuntimeError:
                print("Failed to connect to robot, retrying in 5 seconds")
                time.sleep(5)
        
        if cfg.use_gripper:
            from amiga.drivers.robotiq3f import Robotiq3fGripperModbusTCP

            self.gripper = Robotiq3fGripperModbusTCP()
            self.gripper.connect(hostname=cfg.gripper_ip, port=502)
            print("Gripper connected")

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = cfg.use_gripper
        self.named_configs = {}

        self._load_named_joint_cfgs()

    def _check_required_libraries(self) -> bool:
        ready = True
        try:
            import rtde_control
            import rtde_receive
        except ImportError:
            ready = False
        return ready

    def _check_cfg(self, cfg) -> bool:
        req_keys = ["robot_ip", "gripper_ip", "use_gripper"]
        missing_keys = [key for key in req_keys if key not in cfg]
        return len(missing_keys) == 0

    def _load_named_joint_cfgs(self):
        """Load the named joint configurations from config file."""
        for name, joints in self.cfg.named_joint_cfgs.items():
            js = []
            for j in joints:
                if isinstance(j, str): js.append(eval(j))
                elif isinstance(j, float): js.append(j)
                else: raise ValueError(f"Invalid joint value: {j}")
            
            self.named_configs[name] = np.array(js)
            print(f"Loaded config {name}")

    def get_named_joints_cfg(self, name: str) -> np.ndarray:
        """Get the named joint configuration."""
        assert name in self.named_configs, f"Named config {name} not found"
        cfg = self.named_configs[name]

        if self._use_gripper:
            return cfg
        return cfg[:6]

    def get_num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_eef_speed(self) -> np.ndarray:
        speed = self.r_inter.getActualTCPSpeed()
        # print("Speed: ", [round(s, 1) for s in speed])
        # list with 6 floats, x y z and rx ry rz. X is left, Y is back, Z is up.
        return np.array(speed)
    
    def _get_eef_force(self) -> np.ndarray:
        force = self.r_inter.getActualTCPForce()
        # print("Force: ", [round(f, 1) for f in force])
        # list with 6 floats, x y z and rx ry rz. X is left, Y is back, Z is up.
        return np.array(force)
    
    def _get_eef_pose(self) -> np.ndarray:
        pos = self.r_inter.getActualTCPPose()
        # print("Pos: ", [round(p, 1) for p in pos])
        # list with 6 floats, x y z and rx ry rz. X is left, Y is back, Z is up.
        return np.array(pos)

    def _get_gripper_pos(self) -> float:
        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255
    
    def _get_gripper_object_status(self) -> Tuple[bool, bool, bool]:
        # One bool per finger
        return np.array(self.gripper.has_object())

    def _get_joint_positions(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = np.array(robot_joints)
        return pos
    
    def _get_joint_vel(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        joint_velocities = self.r_inter.getActualQd()  # Current joint velocities
        joint_velocities = np.append(joint_velocities, np.array([0]))

        return joint_velocities

    def go_to_joint_positions(self, joint_positions: np.ndarray, wait: bool = False) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        if self._use_gripper:
            self.robot.moveJ(joint_positions[:6], asynchronous=wait)
            gripper_pos = joint_positions[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        else:
            self.robot.moveJ(joint_positions, asynchronous=wait)
        
        return True

    def servo_joint_positions(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

    def servo_eef_pose_and_gripper(self, pose_and_gripper_angle: np.ndarray) -> None:
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        # print("curr: ", [round(p, 3) for p in self._get_eef_pose()])
        # print("Pose and gripper angle: ", [round(p, 3) for p in pose_and_gripper_angle])

        t_start = self.robot.initPeriod()
        self.robot.servoL(
            pose_and_gripper_angle[:6], velocity, acceleration, dt, lookahead_time, gain
        )

        if self._use_gripper:
            gripper_pos = pose_and_gripper_angle[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
       
        self.robot.waitPeriod(t_start)

    def is_freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observation(self) -> Dict[str, np.ndarray]:
        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_vel()
        ee_force = self._get_eef_force()
        ee_pose_euler = self._get_eef_pose()
        ee_vel = self._get_eef_speed()
        gripper_pos = np.array([joint_pos[-1]])
        gripper_has_obj = self._get_gripper_object_status()
        return {
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "ee_pose_euler": ee_pose_euler,
            "ee_vel": ee_vel,
            "ee_force": ee_force,
            "gripper_position": gripper_pos,
            "gripper_has_object": gripper_has_obj,
        }

    def get_methods(self):
        """This is used to dynamically generate the methods for the ZMQ server."""
        # TODO: use a decorator to generate this automatically
        return {
            "get_num_dofs": None,
            "get_named_joints_cfg": ["name"],
            "get_observation": None,
            "is_freedrive_enabled": None,
            "set_freedrive_mode": ["enable"],
            "servo_joint_positions": ["joint_state"],
            "servo_eef_pose_and_gripper": ["pose_and_gripper_angle"],
            "go_to_joint_positions": ["joint_positions", "wait"],
        }