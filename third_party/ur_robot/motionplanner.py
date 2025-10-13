import mplib
import numpy as np
import trimesh
import os
import re
import time
from typing import Optional, Tuple, List, Dict, Union
from scipy.spatial.transform import Rotation
from .controller import RoboticArm

OPEN = 0.0
CLOSED = 0.8

def debug_pose_to_xyz_rxryrz(mplib_pose):
    # get translation
    p = mplib_pose.get_p()  # or mplib_pose.p
    # get quaternion
    q = mplib_pose.get_q()  # returns [w, x, y, z]
    # convert to [x, y, z, w] order for scipy
    w, x, y, z = q
    quat_scipy = np.array([x, y, z, w])
    # make Rotation object
    r = Rotation.from_quat(quat_scipy)
    rotvec = r.as_rotvec()  # gives (rx, ry, rz) in radians
    return np.concatenate([p.reshape(3,) * 1000, rotvec])

class RealRobotMotionPlanner:
    """
    Motion planning solver for real UR robot with precomputed grasps from output_grasp.npz
    Adapted from simulation-based GraspGenMotionPlanningSolver for real-world use
    """
    
    def __init__(
        self,
        robot: RoboticArm,
        camera_to_robot_transform: np.ndarray = None,
        debug: bool = False,
        joint_vel_limits: float = 0.5,  # Reduced for safety
        joint_acc_limits: float = 0.5,  # Reduced for safety
        grasp_file_path: str = None,
        tcp_offset: float = 0.13,
    ):
        """
        Initialize real robot motion planner
        
        Args:
            robot: RoboticArm instance
            camera_to_robot_transform: 4x4 transformation matrix from camera frame to robot base frame
            debug: Enable debug output
            joint_vel_limits: Maximum joint velocity (rad/s)
            joint_acc_limits: Maximum joint acceleration (rad/s²)
            grasp_file_path: Path to grasp NPZ file
            tcp_offset: TCP offset distance
        """
        self.robot = robot
        self.urdf_path = os.path.join(os.path.dirname(__file__), "robotiq_ur10e", "ur10e_robotiq.urdf")
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.tcp_offset = tcp_offset
        self.debug = debug
        
        # Load precomputed grasps
        if grasp_file_path is None:
            grasp_file_path = os.path.join(os.path.dirname(__file__), "grasps", "apple_grasps.npz")
        self.grasp_data = self.load_grasps(grasp_file_path)
        
        # Setup motion planner
        self.planner = self.setup_planner()
        
        # State tracking
        self.gripper_state = OPEN
        self.current_grasp_idx = 0
        self.use_point_cloud = False

        if camera_to_robot_transform is None:
            raise ValueError("The robot camera calibration matrix must be provided")
        
        if camera_to_robot_transform.shape == (6,):
            # Convert 6D pose to 4x4 transformation matrix
            self.camera_to_robot_transform = self.pose_to_matrix(camera_to_robot_transform)
            print(f"Camera-robot calibration loaded from 6D pose: {camera_to_robot_transform}")
        elif camera_to_robot_transform.shape == (4, 4):
            # Already a 4x4 matrix
            self.camera_to_robot_transform = camera_to_robot_transform
            print("Camera-robot calibration loaded from 4x4 matrix")
        else:
            raise ValueError(f"Transform must be 6D pose [x,y,z,rx,ry,rz] or 4x4 matrix, got shape: {camera_to_robot_transform.shape}")
        
        print("Real robot motion planner initialized successfully")

    def load_grasps(self, grasp_file_path: str) -> Dict:
        """
        Load precomputed grasps from NPZ file
        """
        if not os.path.exists(grasp_file_path):
            raise FileNotFoundError(f"Grasp file not found: {grasp_file_path}")
        
        grasp_data = np.load(grasp_file_path)
        
        # Check if the grasps are relative
        if 'is_relative' in grasp_data and grasp_data['is_relative']:
            print(f"Loaded {grasp_data['num_grasps']} RELATIVE grasps from {grasp_file_path}")
            # Adjust keys to be consistent
            if 'relative_poses' in grasp_data:
                grasp_data = dict(grasp_data)  # Make it mutable
                grasp_data['poses'] = grasp_data['relative_poses']
                grasp_data['positions'] = grasp_data['relative_positions']
                grasp_data['quaternions'] = grasp_data['relative_quaternions']
        else:
            print(f"Loaded {grasp_data['num_grasps']} ABSOLUTE grasps from {grasp_file_path}")

        print(f"Best grasp confidence: {grasp_data['confidences'][0]:.4f}")
        print(f"Confidence range: [{grasp_data['confidence_range'][0]:.4f}, {grasp_data['confidence_range'][1]:.4f}]")
        
        return grasp_data
    
    def transform_pose_camera_to_robot(self, camera_pose: Union[np.ndarray, List]) -> np.ndarray:
        """
        Transform object pose from camera frame to robot base frame
        
        Args:
            camera_pose: 6D pose [x, y, z, rx, ry, rz], 7D pose [x, y, z, qx, qy, qz, qw], 
                        or 4x4 transformation matrix
            
        Returns:
            6D pose in robot frame [x, y, z, rx, ry, rz]
        """
        camera_pose = np.array(camera_pose)
        
        # Check if input is already a 4x4 transformation matrix
        if camera_pose.shape == (4, 4):
            camera_matrix = camera_pose
        else:
            # Convert 6D or 7D pose to 4x4 matrix
            camera_matrix = self.pose_to_matrix(camera_pose)
        
        # Transform to robot frame: Robot_Pose = Camera_to_Robot * Camera_Pose
        robot_matrix = self.camera_to_robot_transform @ camera_matrix
        
        # Convert back to 6D pose
        robot_pose = self.matrix_to_pose(robot_matrix, format='rotvec')
        
        if self.debug:
            if camera_pose.shape == (4, 4):
                print(f"Camera pose (matrix):\n{camera_pose}")
            else:
                print(f"Camera pose: {camera_pose}")
            print(f"Robot pose: {robot_pose}")
        
        return robot_pose

    def transform_pose_robot_to_camera(self, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform pose from robot base frame to camera frame (inverse transform)
        
        Args:
            robot_pose: 6D pose in robot frame [x, y, z, rx, ry, rz]
            
        Returns:
            6D pose in camera frame [x, y, z, rx, ry, rz]
        """
        # Convert robot pose to 4x4 matrix
        robot_matrix = self.pose_to_matrix(robot_pose)
        
        # Transform to camera frame: Camera_Pose = Robot_to_Camera * Robot_Pose
        robot_to_camera_transform = np.linalg.inv(self.camera_to_robot_transform)
        camera_matrix = robot_to_camera_transform @ robot_matrix
        
        # Convert back to 6D pose
        camera_pose = self.matrix_to_pose(camera_matrix, format='rotvec')
        
        return camera_pose

    def setup_planner(self) -> mplib.Planner:
        """
        Setup mplib planner for real robot
        """
        # Generate missing convex hull files
        self._generate_missing_convex_meshes()
        
        # Create SRDF path if not provided
        self.srdf_path = self.urdf_path.replace(".urdf", ".srdf")

        # Define the 6 arm joints we want to control (from URDF analysis)
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'left_outer_knuckle_joint', 'left_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint']
        
        # Define relevant link names (you can include all or just the important ones)
        link_names = [
            'base_link',
            'base_link_inertia',
            'base',
            'shoulder_link',
            'upper_arm_link',
            'forearm_link',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
            'flange',
            'tool0',
            'robotiq_arg2f_base_link',
            'eef',
            'left_outer_knuckle',
            'left_inner_knuckle',
            'right_outer_knuckle',
            'right_inner_knuckle',
            'left_outer_finger',
            'right_outer_finger',
            'left_inner_finger',
            'right_inner_finger',
            'left_inner_finger_pad',
            'right_inner_finger_pad'
        ]

        planner = mplib.Planner(
            urdf=self.urdf_path,
            srdf=self.srdf_path if os.path.exists(self.srdf_path) else None,
            user_joint_names=joint_names,
            user_link_names=link_names,
            move_group="eef",  # UR robot end-effector
            joint_vel_limits=np.ones(6) * self.joint_vel_limits,
            joint_acc_limits=np.ones(6) * self.joint_acc_limits,
        )
        print("Motion planner initialized with collision detection enabled")
        
        return planner

    def _generate_missing_convex_meshes(self):
        """
        Automatically generate convex hull files for all .stl files in URDF
        """
        with open(self.urdf_path, 'r') as f:
            urdf_content = f.read()
        
        urdf_dir = os.path.dirname(self.urdf_path)
        stl_pattern = r'filename="([^"]*\.stl)"'
        stl_matches = re.findall(stl_pattern, urdf_content)
        
        generated_count = 0
        for stl_mesh_path in stl_matches:
            if '.convex.stl' in stl_mesh_path:
                continue
                
            if not os.path.isabs(stl_mesh_path):
                full_stl_path = os.path.join(urdf_dir, stl_mesh_path)
            else:
                full_stl_path = stl_mesh_path
            
            convex_path = full_stl_path.replace('.stl', '.stl.convex.stl')
            
            if os.path.exists(full_stl_path) and not os.path.exists(convex_path):
                if self.debug:
                    print(f"Generating convex hull file: {os.path.basename(convex_path)}")
                
                try:
                    mesh = trimesh.load(full_stl_path)
                    if hasattr(mesh, 'geometry') and hasattr(mesh.geometry, 'values'):
                        all_vertices = np.vstack([geom.vertices for geom in mesh.geometry.values() if hasattr(geom, 'vertices')])
                        convex_mesh = trimesh.convex.convex_hull(all_vertices)
                    else:
                        convex_mesh = mesh.convex_hull
                    
                    os.makedirs(os.path.dirname(convex_path), exist_ok=True)
                    convex_mesh.export(convex_path)
                    generated_count += 1
                except Exception as e:
                    print(f"Warning: Failed to generate convex hull for {stl_mesh_path}: {e}")
        
        if generated_count > 0:
            print(f"Generated {generated_count} convex hull files")

    def get_current_joint_positions(self) -> np.ndarray:
        """Get current joint positions from real robot"""
        state = self.robot.get_state()
        return state['ActualQ']

    def get_current_tcp_pose(self) -> np.ndarray:
        """Get current TCP pose from real robot [x, y, z, rx, ry, rz]"""
        state = self.robot.get_state()
        return state['ActualTCPPose']

    def pose_to_matrix(self, pose: Union[np.ndarray, List]) -> np.ndarray:
        """Convert 6D pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix"""
        pose = np.array(pose)
        if len(pose) == 6:
            pos = pose[:3]
            rot_vec = pose[3:]
            rot = Rotation.from_rotvec(rot_vec)
            matrix = np.eye(4)
            matrix[:3, :3] = rot.as_matrix()
            matrix[:3, 3] = pos
            return matrix
        elif len(pose) == 7:
            # [x, y, z, qx, qy, qz, qw]
            pos = pose[:3]
            quat = pose[3:]
            rot = Rotation.from_quat(quat)
            matrix = np.eye(4)
            matrix[:3, :3] = rot.as_matrix()
            matrix[:3, 3] = pos
            return matrix
        else:
            raise ValueError(f"Invalid pose format: {pose}")

    def matrix_to_pose(self, matrix: np.ndarray, format='rotvec') -> np.ndarray:
        """Convert 4x4 transformation matrix to 6D or 7D pose"""
        pos = matrix[:3, 3]
        rot = Rotation.from_matrix(matrix[:3, :3])
        
        if format == 'rotvec':
            return np.concatenate([pos, rot.as_rotvec()])
        elif format == 'quat':
            return np.concatenate([pos, rot.as_quat()])
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_relative_grasp_pose(self, grasp_idx: int = 0) -> np.ndarray:
        """
        Get relative grasp pose by index (0 = best grasp)
        Returns 4x4 transformation matrix relative to object frame
        """
        if grasp_idx >= self.grasp_data['num_grasps']:
            raise IndexError(f"Grasp index {grasp_idx} out of range (max: {self.grasp_data['num_grasps']-1})")
        
        # FoundationPose correction transformation
        T_corr = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, self.tcp_offset],
            [0, 0, 0, 1],
        ])

        grasp_matrix = self.grasp_data['poses'][grasp_idx] @ T_corr
        return grasp_matrix

    def get_world_grasp_pose(self, object_pose_camera: np.ndarray, grasp_idx: int = 0, coordinate_frame: str = 'camera') -> np.ndarray:
        """
        Calculate world-frame grasp pose from object pose and relative grasp
        
        Args:
            object_pose_camera: 6D object pose [x, y, z, rx, ry, rz] 
            grasp_idx: Index of grasp to use
            coordinate_frame: 'camera' if object_pose is in camera frame, 'robot' if already in robot frame
            
        Returns:
            6D world grasp pose in robot frame [x, y, z, rx, ry, rz]
        """
        # Transform object pose to robot frame if needed
        if coordinate_frame == 'camera':
            object_pose_robot = self.transform_pose_camera_to_robot(object_pose_camera)
        else:
            object_pose_robot = object_pose_camera
        
        # Convert object pose to transformation matrix
        object_matrix = self.pose_to_matrix(object_pose_robot)
        
        # Get relative grasp transformation
        relative_grasp_matrix = self.get_relative_grasp_pose(grasp_idx)
        
        # Combine: World_Grasp = Object_World * Relative_Grasp
        world_grasp_matrix = object_matrix @ relative_grasp_matrix
        
        # Convert back to 6D pose
        world_grasp_pose = self.matrix_to_pose(world_grasp_matrix, format='rotvec')
        
        if self.debug:
            print(f"Object pose (camera): {object_pose_camera}")
            print(f"Object pose (robot): {object_pose_robot}")
            print(f"Grasp pose (robot): {world_grasp_pose}")
        
        return world_grasp_pose

    def get_best_grasps(self, object_pose_camera: np.ndarray, n_grasps: int = 5, coordinate_frame: str = 'camera') -> List[Tuple[np.ndarray, float]]:
        """
        Get the top N best grasps sorted by confidence
        
        Args:
            object_pose_camera: 6D object pose in camera or robot frame
            n_grasps: Number of best grasps to return
            coordinate_frame: 'camera' or 'robot'
            
        Returns:
            List of (pose_in_robot_frame, confidence) tuples
        """
        n_grasps = min(n_grasps, self.grasp_data['num_grasps'])
        grasp_poses = []
        
        for i in range(n_grasps):
            pose = self.get_world_grasp_pose(object_pose_camera, i, coordinate_frame)
            confidence = self.grasp_data['confidences'][i]
            grasp_poses.append((pose, confidence))
        
        return grasp_poses

    def plan_to_pose_rrt(self, target_pose: np.ndarray, start_joints: Optional[np.ndarray] = None) -> Dict:
        """
        Plan path to target pose using RRT-Connect algorithm
        
        Args:
            target_pose: 6D target pose [x, y, z, rx, ry, rz]
            start_joints: Starting joint configuration (uses current if None)
            
        Returns:
            Planning result dictionary
        """
        if start_joints is None:
            start_joints = self.get_current_joint_positions()
        
        # Convert pose to mplib.Pose format
        from mplib.pymp import Pose
        pos = target_pose[:3]
        rot_vec = target_pose[3:]
        rot = Rotation.from_rotvec(rot_vec)
        quat = rot.as_quat()  # [x, y, z, w]
        
        # Create mplib.Pose object
        goal_pose = Pose(pos, quat)
        
        if self.debug:
            print(f"Planning RRT to pose: {target_pose}")
            print(f"From joints: {start_joints}")
        
        result = self.planner.plan_pose(
            goal_pose=goal_pose,
            current_qpos=start_joints,
            time_step=0.008,  # 125Hz control frequency
            wrt_world=True,
            planning_time=5.0,  # Give more time for planning
            rrt_range=0.1,
            verbose=self.debug
        )
        
        if self.debug:
            print(f"RRT planning result: {result['status']}")
        
        return result

    def plan_to_joints_rrt(self, target_joints: np.ndarray, start_joints: Optional[np.ndarray] = None) -> Dict:
        """
        Plan path to target joint configuration using RRT
        """
        if start_joints is None:
            start_joints = self.get_current_joint_positions()
        
        if self.debug:
            print(f"Planning RRT to joints: {target_joints}")
            print(f"From joints: {start_joints}")
            
        result = self.planner.plan_qpos(
            goal_qposes=[target_joints],  # Note: this expects a list
            current_qpos=start_joints,
            time_step=0.008,
            planning_time=5.0,
            rrt_range=0.1,
            verbose=self.debug
        )
        
        if self.debug:
            print(f"Joint RRT planning result: {result['status']}")
        
        return result

    def plan_screw_motion(self, target_pose: np.ndarray, start_joints: Optional[np.ndarray] = None) -> Dict:
        """
        Plan straight-line motion in Cartesian space
        """
        if start_joints is None:
            start_joints = self.get_current_joint_positions()
        
        # Convert pose to mplib.Pose format
        from mplib.pymp import Pose
        pos = target_pose[:3]
        rot_vec = target_pose[3:]
        rot = Rotation.from_rotvec(rot_vec)
        quat = rot.as_quat()
        
        goal_pose = Pose(pos, quat)
        
        if self.debug:
            print(f"Planning screw motion to pose: {target_pose}")
            
        result = self.planner.plan_screw(   # FIXME: Please check which coordinate system is used here
            goal_pose,
            start_joints,
            time_step=0.008,
            # wrt_world=True,
            # verbose=self.debug
        ) # goal_pose = [-951.8541469909146, -53.174801314369425, 270.3962424086691, 0.8467532487376251, -1.438695340652482, 1.4228045503522344]
        
        if self.debug:
            print(f"Screw motion planning result: {result['status']}")
        
        return result

    def execute_trajectory(self, trajectory: np.ndarray, execution_time: float = None, 
                          gripper_action: Optional[float] = None) -> bool:
        """
        Execute planned trajectory on real robot
        
        Args:
            trajectory: Joint trajectory array [n_waypoints, 6]
            execution_time: Total time to execute trajectory
            gripper_action: Gripper position (0-1) during execution
            
        Returns:
            True if execution successful
        """
        n_waypoints = trajectory.shape[0]
        
        if execution_time is None:
            # Estimate execution time based on joint velocities
            execution_time = max(n_waypoints * 0.1, 2.0)  # Minimum 2 seconds
        
        # Calculate timing for each waypoint
        waypoint_times = np.linspace(0, execution_time, n_waypoints)
        start_time = time.time()
        
        print(f"Executing trajectory with {n_waypoints} waypoints over {execution_time:.2f}s")
        
        for i, (waypoint, relative_time) in enumerate(zip(trajectory, waypoint_times)):
            # Schedule joint waypoint
            absolute_time = start_time + relative_time
            self.robot.exec_joint_action(waypoint, absolute_time)
            
            # Command gripper if specified
            if gripper_action is not None:
                self.robot.command_gripper(gripper_action)
            
            # Small delay to prevent overwhelming the controller
            if i < len(trajectory) - 1:
                time.sleep(0.01)
        
        # Wait for trajectory completion
        time.sleep(execution_time + 0.5)
        return True

    def move_to_pose(self, target_pose: np.ndarray, use_rrt: bool = True, 
                    execution_time: float = None, gripper_action: Optional[float] = None) -> bool:
        """
        Plan and execute motion to target pose
        
        Args:
            target_pose: 6D target pose [x, y, z, rx, ry, rz]
            use_rrt: Use RRT planning (True) or screw motion (False)
            execution_time: Time to execute motion
            gripper_action: Gripper position during motion
            
        Returns:
            True if successful
        """
        print(f"Planning motion to pose: {target_pose}")
        
        # Plan trajectory
        if use_rrt:
            result = self.plan_to_pose_rrt(target_pose)
        else:
            result = self.plan_screw_motion(target_pose)
        
        if result["status"] != "Success":
            print(f"Motion planning failed: {result['status']}")
            return False
        
        print(f"Planning successful! Trajectory has {result['position'].shape[0]} waypoints")
        
        # Execute trajectory
        return self.execute_trajectory(result["position"], execution_time, gripper_action)

    def move_to_joints(self, target_joints: np.ndarray, execution_time: float = None,
                      gripper_action: Optional[float] = None) -> bool:
        """
        Plan and execute motion to target joint configuration
        """
        print(f"Planning motion to joints: {target_joints}")
        
        result = self.plan_to_joints_rrt(target_joints)
        
        if result["status"] != "Success":
            print(f"Joint motion planning failed: {result['status']}")
            return False
        
        print(f"Planning successful! Trajectory has {result['position'].shape[0]} waypoints")
        
        return self.execute_trajectory(result["position"], execution_time, gripper_action)

    def open_gripper(self, wait_time: float = 1.0) -> bool:
        """
        Open the gripper
        """
        print("Opening gripper...")
        self.gripper_state = OPEN
        self.robot.command_gripper(self.gripper_state)
        time.sleep(wait_time)
        return True

    def close_gripper(self, grip_force: float = CLOSED, wait_time: float = 2.0) -> bool:
        """
        Close the gripper
        
        Args:
            grip_force: Gripper closing force (0.0 to 1.0)
            wait_time: Time to wait for gripper to close
        """
        print(f"Closing gripper with force {grip_force}...")
        self.gripper_state = grip_force
        self.robot.command_gripper(self.gripper_state, speed=100, force=50)
        time.sleep(wait_time)
        return True

    def close(self):
        """
        Clean up resources
        """
        if self.planner is not None:
            del self.planner
            self.planner = None
        print("Motion planner closed")
