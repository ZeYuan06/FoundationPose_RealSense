import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from third_party.ur_robot.rtde_interpolation_controller import RTDEInterpolationController

class RoboticArm:
    """
    A high-level interface for controlling a robotic arm and gripper.

    This class provides an abstraction layer to simplify interaction with the robotic arm and gripper, encapsulating the complexities of the Real-Time Data Exchange interpolation controller interface. It is designed to facilitate seamless and efficient control of the robotic hardware.
    """

    def __init__(self,
                robot_ip,
                frequency=125,
                max_pos_speed=0.25,
                max_rot_speed=0.6,
                tcp_offset=0.13,
                init_joints=True,
                use_gripper=True,
                gripper_port=52345,
                shm_manager: SharedMemoryManager=None,
        ):
        """
        Initializes the RoboticArm controller.

        Args:
            robot_ip (str): The IP address of the robot.
            frequency (int, optional): The control frequency. Defaults to 125.
            max_pos_speed (float, optional): The maximum positional speed. Defaults to 0.25.
            max_rot_speed (float, optional): The maximum rotational speed. Defaults to 0.6.
            tcp_offset (float, optional): The TCP offset. Defaults to 0.13.
            init_joints (bool, optional): Whether to initialize joints to a predefined position. Defaults to True.
            use_gripper (bool, optional): Whether to use the gripper. Defaults to True.
            gripper_port (int, optional): The port for the gripper. Defaults to 63352.
            shm_manager (SharedMemoryManager, optional): A shared memory manager. If None, a new one is created. Defaults to None.
        """

        if shm_manager is None:
            self._shm_manager = SharedMemoryManager()
            self._shm_manager.start()
        else:
            self._shm_manager = shm_manager

        j_init = np.array([0, -90, 90, -90, -90, 0]) / 180 * np.pi if init_joints else None
        cube_diag = np.linalg.norm([1, 1, 1])

        self.controller = RTDEInterpolationController(
            shm_manager=self._shm_manager,
            robot_ip=robot_ip,
            frequency=frequency,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=float(max_pos_speed * cube_diag),
            max_rot_speed=float(max_rot_speed * cube_diag),
            tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
            joints_init=j_init,
            use_gripper=use_gripper,
            gripper_port=gripper_port,
            get_max_k=2,
        )

    def start(self, wait=True):
        """
        Starts the robot controller.

        Args:
            wait (bool, optional): Whether to wait for the controller to be ready. Defaults to True.
        """
        self.controller.start(wait=wait)

    def stop(self, wait=True):
        """
        Stops the robot controller and cleans up resources.

        Args:
            wait (bool, optional): Whether to wait for the controller to stop. Defaults to True.
        """
        self.controller.stop(wait=wait)
        if self._shm_manager is not None:
            self._shm_manager.shutdown()
            self._shm_manager = None

    def __enter__(self):
        """
        Context manager entry point. Starts the controller.
        """
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Stops the controller.
        """
        self.stop()

    @property
    def is_ready(self):
        """
        Checks if the robot controller is ready for new commands.

        Returns:
            bool: True if the controller is ready, False otherwise.
        """
        return self.controller.is_ready
    
    def get_state(self) -> dict:
        """
        Gets the current state of the robot.

        Returns:
            dict: A dictionary containing the robot's state information.
        """
        return self.controller.get_state()
    
    def exec_action(self, action: np.ndarray, target_time: float):
        """
        Executes a Cartesian pose action.

        Args:
            action (np.ndarray): A 7-element array containing the target pose (6 elements) and gripper position (1 element).
            target_time (float): The target time for reaching the pose.
        """
        assert self.is_ready
        assert action.shape == (7,)

        pose = action[:6]
        gripper_pos = action[6]

        # Use servoJ for immediate Cartesian control
        self.controller.servoJ(pose, duration=0.1)
        self.controller.command_gripper(gripper_pos)

    def exec_joint_action(self, joint_action: np.ndarray, target_time: float):
        """
        Executes a joint space action.

        Args:
            joint_action (np.ndarray): A 6-element array of target joint positions.
            target_time (float): The target time for reaching the joint positions.
        """
        assert self.is_ready
        assert joint_action.shape == (6,)

        # Use servoJ for immediate joint control
        self.controller.servoJ(joint_action, duration=0.1)

    def command_gripper(self, gripper_pos: float, speed=255, force=100):
        """
        Commands the gripper to a specific position.

        Args:
            gripper_pos (float): The target gripper position.
            speed (int, optional): The speed of the gripper movement. Defaults to 255.
            force (int, optional): The force of the gripper movement. Defaults to 100.
        """
        assert self.is_ready
        self.controller.command_gripper(gripper_pos, speed, force)


def main():
    """
    Test function to verify robotic arm connectivity and basic functionality.
    """
    # Robot configuration - update these values for your specific robot
    robot_ip = "192.168.50.144"  # Replace with your robot's IP address
    
    print("Initializing robotic arm...")

    # Create and start the robotic arm controller
    with RoboticArm(
        robot_ip=robot_ip,
        frequency=125,
        max_pos_speed=0.1,  # Slower speed for safety during testing
        max_rot_speed=0.3,
        tcp_offset=0.13,
        init_joints=True,
        use_gripper=True,
        gripper_port=63352
    ) as robot:
        # Get initial robot state
        initial_state = robot.get_state()
        print("Initial robot state received")
        print(f"Current TCP pose: {initial_state.get('ActualTCPPose', 'Not available')}")
        print(f"Current joint positions: {initial_state.get('ActualQ', 'Not available')}")
        
        # Test 1: Send a simple gripper command
        print("Test 1: Testing gripper control...")
        robot.command_gripper(0.5, speed=100, force=50)  # Half open
        time.sleep(2.0)
        
        robot.command_gripper(0.0, speed=100, force=50)  # Fully open
        time.sleep(2.0)
        
        print("Gripper test completed")
        
        # Test 2: Send a test Cartesian pose action
        print("Test 2: Testing Cartesian pose control...")
        
        # Get current TCP pose and make small movement
        test_pose = np.array([0, -80, 100, -100, -100, 0]) / 180 * np.pi
        
        # Create action with pose + gripper
        test_action = np.concatenate([test_pose, [0.0]])  # 6D pose + gripper
        target_time = time.time() + 3.0
        
        robot.exec_action(test_action, target_time)
        print("Cartesian pose command sent")
        time.sleep(4.0)  # Wait for execution
        
        # Test 3: Send a test joint action
        print("Test 3: Testing joint space control...")
        
        # Get current joint positions and make small movement
        current_joints = initial_state["ActualQ"]
        joint_action = current_joints.copy()
        joint_action[5] += 0.1  # Small wrist rotation
        target_time = time.time() + 3.0
        
        robot.exec_joint_action(joint_action, target_time)
        print("Joint space command sent")
        time.sleep(4.0)  # Wait for execution
        
        # Get final robot state
        final_state = robot.get_state()
        print("Final robot state received")
        print("All tests completed successfully!")
    
    print("Test completed")


if __name__ == "__main__":
    main()
