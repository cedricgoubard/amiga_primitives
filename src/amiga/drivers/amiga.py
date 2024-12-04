from typing import Dict, Tuple
import time

import numpy as np
from omegaconf import OmegaConf

from amiga.drivers.zmq import ZMQBackendObject


import numpy as np

def rpy2rv(roll, pitch, yaw):
    alpha = yaw
    beta = pitch
    gamma = roll
    
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    
    # Compute rotation matrix elements
    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg
    
    # Calculate theta (rotation angle)
    trace = r11 + r22 + r33
    theta = np.arccos((trace - 1) / 2)
    
    # Handle small theta to avoid division by zero
    sth = np.sin(theta)
    if np.isclose(sth, 0):
        kx, ky, kz = 0.0, 0.0, 0.0  # Define an arbitrary axis in case of no rotation
    else:
        kx = (r32 - r23) / (2 * sth)
        ky = (r13 - r31) / (2 * sth)
        kz = (r21 - r12) / (2 * sth)
    
    # Rotation vector
    rv = np.array([theta * kx, theta * ky, theta * kz])
    return rv

def rv2rpy(rx, ry, rz):
    theta = np.sqrt(rx**2 + ry**2 + rz**2)
    
    # Handle zero rotation case
    if np.isclose(theta, 0):
        return np.array([0.0, 0.0, 0.0])
    
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    cth = np.cos(theta)
    sth = np.sin(theta)
    vth = 1 - cth
    
    # Compute rotation matrix elements
    r11 = kx * kx * vth + cth
    r12 = kx * ky * vth - kz * sth
    r13 = kx * kz * vth + ky * sth
    r21 = kx * ky * vth + kz * sth
    r22 = ky * ky * vth + cth
    r23 = ky * kz * vth - kx * sth
    r31 = kx * kz * vth - ky * sth
    r32 = ky * kz * vth + kx * sth
    r33 = kz * kz * vth + cth
    
    # Compute beta (pitch)
    beta = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
    
    # Handle gimbal lock
    if beta > np.radians(89.99):
        beta = np.radians(89.99)
        alpha = 0
        gamma = np.arctan2(r12, r22)
    elif beta < np.radians(-89.99):
        beta = np.radians(-89.99)
        alpha = 0
        gamma = -np.arctan2(r12, r22)
    else:
        cb = np.cos(beta)
        alpha = np.arctan2(r21 / cb, r11 / cb)
        gamma = np.arctan2(r32 / cb, r33 / cb)
    
    # Roll (gamma), Pitch (beta), Yaw (alpha)
    rpy = np.array([gamma, beta, alpha])
    return rpy


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

    def _reload_ur_program_if_not_running(self):
        while not self.robot.isProgramRunning():
            print("UR program not running, re-uploading")
            self.robot.reuploadScript()
            time.sleep(0.3)

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

    def get_camera_tf(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array([0.14900713, 0.06125623, 0.01208498]),
            np.array([
                [-0.64635911, -0.69014093, -0.32546181],
                [ 0.76197034, -0.6063051,  -0.22758585],
                [-0.04026285, -0.39509444,  0.91775775]
                ])
        )

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
        pos = np.array(self.r_inter.getActualTCPPose())
        # list with 6 floats, x y z and rx ry rz. X is left, Y is back, Z is up. ROTATION ANGLES!
        pos[3:6] = rv2rpy(*pos[3:6])
        return pos

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

    def go_to_eef_pose(self, eef_pose: np.ndarray, gripper_position: float = None, wait: bool = False) -> None:
        """Command the leader robot to a given state.

        Args:
            eef_pose (np.ndarray): The state to command the leader robot to. Rotation is RPY in this order with extrinsic axes.
            gripper_position (float): The position of the gripper.
        """
        self._reload_ur_program_if_not_running()

        # UR expects rotation vector, not RPY
        eef_pose[3:6] = rpy2rv(eef_pose[3], eef_pose[4], eef_pose[5])

        if self._use_gripper:
            self.robot.moveL(pose=eef_pose[:6], asynchronous=(not wait))
            if gripper_position is not None:
                gripper_pos = gripper_position * 255
                self.gripper.move(gripper_pos, 255, 10)
        else:
            self.robot.moveL(eef_pose, asynchronous=(not wait))

    def go_to_eef_position_default_orientation(self, eef_position: np.ndarray, gripper_position: float = None, wait: bool = False) -> None:
        self._reload_ur_program_if_not_running()
        
        default_rpy = [np.pi/2, -np.pi/4, 0.0]  # Gripper facing forward
        eef_pose = np.append(eef_position, rpy2rv(*default_rpy))
        print("EEF pose: ", eef_pose)
        if self._use_gripper:
            self.robot.moveJ_IK(pose=eef_pose[:6], asynchronous=(not wait))
            if gripper_position is not None:
                gripper_pos = gripper_position * 255
                self.gripper.move(gripper_pos, 255, 10)
        else:
            self.robot.moveJ_IK(eef_pose, asynchronous=(not wait))
        
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

    def close_gripper(self):
        if self._use_gripper:
            self.gripper.close()

    def open_gripper(self):
        if self._use_gripper:
            self.gripper.open()

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
            "go_to_eef_pose": ["eef_pose", "gripper_position", "wait"],
            "go_to_eef_position_default_orientation": ["eef_position", "gripper_position", "wait"],
            "close_gripper": None,
            "open_gripper": None,
            "get_camera_tf": None,
        }