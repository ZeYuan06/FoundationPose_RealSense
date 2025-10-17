import numpy as np
import time
import sys
import os
from typing import Dict

# Import the real robot motion planner and robot controller
from .motionplanner import RealRobotMotionPlanner as MotionPlanner
from .controller import RoboticArm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

def visualize_poses(poses: Dict[str, np.ndarray], title: str = "Pose Visualization", axis_length: float = 0.1):
    """
    Visualizes a dictionary of 4x4 transformation matrices as 3D coordinate frames.

    Args:
        poses (Dict[str, np.ndarray]): A dictionary where keys are labels (e.g., "camera")
                                       and values are 4x4 transformation matrices.
        title (str): The title of the plot.
        axis_length (float): The length of the coordinate frame axes to draw.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    all_points = []

    for name, matrix in poses.items():
        if not isinstance(matrix, np.ndarray) or matrix.shape != (4, 4):
            print(f"Warning: Skipping '{name}' due to invalid format. Expected 4x4 np.ndarray, got {type(matrix)} with shape {matrix.shape if isinstance(matrix, np.ndarray) else 'N/A'}.")
            continue

        # Origin of the frame
        origin = matrix[:3, 3]
        all_points.append(origin)

        # Axis directions
        x_axis = matrix[:3, 0]
        y_axis = matrix[:3, 1]
        z_axis = matrix[:3, 2]

        # Draw the axes
        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=axis_length, color='r', label=f'{name}_X')
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=axis_length, color='g', label=f'{name}_Y')
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=axis_length, color='b', label=f'{name}_Z')

        # Add a label
        ax.text(origin[0], origin[1], origin[2], f'  {name}', color='k')

    # Auto-scale plot
    if all_points:
        points = np.array(all_points)
        max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                              points[:, 1].max()-points[:, 1].min(), 
                              points[:, 2].max()-points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def solve_pickbox_with_grasp_real_robot(
    robot_ip: str,
    object_pose: np.ndarray,
    grasp_file_path: str = None,
    grasp_idx: int = 0,
    debug: bool = False,
    approach_distance: float = 0.1,
    lift_height: float = 0.1,
    return_to_home: bool = True
) -> bool:
    """
    Solve pick and place task using precomputed grasp on real robot
    
    Args:
        robot_ip: IP address of the UR robot
        object_pose: 6D object pose [x, y, z, rx, ry, rz] in world frame
        grasp_file_path: Path to grasp NPZ file (default: use bleach_cleanser_grasps.npz)
        grasp_idx: Index of the grasp to use (0 = best grasp)
        debug: Whether to enable debug mode
        approach_distance: Distance to approach from (meters)
        lift_height: Height to lift object (meters)
        return_to_home: Whether to return to initial position after picking
    
    Returns:
        bool: True if task completed successfully, False otherwise
    """
    if grasp_file_path is None:
        grasp_file_path = os.path.join(os.path.dirname(__file__), "grasps", "flask_grasps.npz")

    print(f"Starting pick task with grasp {grasp_idx}")
    print(f"Object pose: {object_pose}")
    print(f"Grasp file: {grasp_file_path}")
    
    # Initialize robot
    with RoboticArm(
        robot_ip=robot_ip,
        frequency=125,
        max_pos_speed=0.1,  # Safe speeds for picking
        max_rot_speed=0.3,
        tcp_offset=0.13,
        init_joints=True,
        use_gripper=True,
        gripper_port=63352
    ) as robot:
        
        # Wait for robot to be ready
        print("Waiting for robot to be ready...")
        while not robot.get_state():
            time.sleep(0.1)
        
        print("Robot ready!")
        
        # Initialize motion planner
        # object_pose_camera = np.array([-0.9588, -0.5602,  0.4839, -2.1478,  0.5586, -0.4527])
        object_pose_camera = np.array([-1.1940, -0.5883,  0.4981, -2.0145,  0.6835, -0.6615])
        planner = MotionPlanner(
            robot=robot,
            camera_to_robot_transform=object_pose_camera,
            debug=debug,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
            grasp_file_path=grasp_file_path,
            tcp_offset=0.20
        )

        if debug:
            print("Visualizing coordinate frames for debugging...")
            
            # 1. Get the grasp pose in the robot's frame
            grasp_pose_robot_frame = planner.get_world_grasp_pose(object_pose, grasp_idx, coordinate_frame='camera')
            
            # 2. Get the object pose in the robot's frame
            object_pose_robot_frame = planner.transform_pose_camera_to_robot(object_pose)

            # 3. Collect all poses as 4x4 matrices
            poses_to_visualize = {
                "robot_base": np.eye(4),
                "camera": planner.camera_to_robot_transform,
                "object": planner.pose_to_matrix(object_pose_robot_frame),
                "grasp_target": planner.pose_to_matrix(grasp_pose_robot_frame)
            }
            
            # 4. Call the visualization function
            visualize_poses(poses_to_visualize, title="Initial Setup Visualization")
        
        # Get initial TCP pose for returning home later
        initial_tcp_pose = planner.get_current_tcp_pose()
        print(f"Initial TCP pose: {initial_tcp_pose}")

        # Get precomputed grasp pose in world coordinates
        grasp_pose = planner.get_world_grasp_pose(object_pose, grasp_idx)
        confidence = planner.grasp_data['confidences'][grasp_idx]
        print(f"Using grasp {grasp_idx} with confidence: {confidence:.4f}")
        print(f"Target world grasp pose: {grasp_pose}")

        # -------------------------------------------------------------------------- #
        # Stage 1: Open gripper
        # -------------------------------------------------------------------------- #
        print("Stage 1: Opening gripper")
        if not planner.open_gripper():
            print("Failed to open gripper")
            return False

        # -------------------------------------------------------------------------- #
        # Stage 2: Move to pre-grasp position
        # -------------------------------------------------------------------------- #
        print("Stage 2: Moving to pre-grasp position")
        # Create approach pose (move back along Z-axis)
        approach_pose = grasp_pose.copy()
        approach_pose[2] += approach_distance  # Move back in Z direction
        print(f"The approach pose is {approach_pose}")
        return
        
        print(f"Approach pose: {approach_pose}")
        success = planner.move_to_pose(approach_pose, use_rrt=False, execution_time=4.0)
        if not success:
            print("Pre-grasp position planning failed")
            planner.close()
            return False
        return

        # -------------------------------------------------------------------------- #
        # Stage 3: Move precisely to grasp position using straight-line motion
        # -------------------------------------------------------------------------- #
        print("Stage 3: Moving to grasp position")
        success = planner.move_to_pose(grasp_pose, use_rrt=False, execution_time=3.0)
        if not success:
            print("Grasp position planning failed")
            planner.close()
            return False

        # -------------------------------------------------------------------------- #
        # Stage 4: Close gripper
        # -------------------------------------------------------------------------- #
        print("Stage 4: Closing gripper")
        if not planner.close_gripper(grip_force=0.8, wait_time=2.0):
            print("Failed to close gripper")
            planner.close()
            return False

        # -------------------------------------------------------------------------- #
        # Stage 5: Lift object
        # -------------------------------------------------------------------------- #
        print("Stage 5: Lifting object")
        lift_pose = grasp_pose.copy()
        lift_pose[2] += lift_height  # Lift up
        
        print(f"Lift pose: {lift_pose}")
        success = planner.move_to_pose(lift_pose, use_rrt=False, execution_time=3.0)
        if not success:
            print("Lift motion planning failed")
            planner.close()
            return False

        # -------------------------------------------------------------------------- #
        # Stage 6: Return to home position (optional)
        # -------------------------------------------------------------------------- #
        if return_to_home:
            print("Stage 6: Moving to home position")
            success = planner.move_to_pose(initial_tcp_pose, use_rrt=False, execution_time=5.0)
            if not success:
                print("Home position planning failed")
                planner.close()
                return False
            
            # Optional: Release object at home position
            # print("Stage 7: Releasing object")
            # planner.open_gripper()
        
        print("Pick task completed successfully!")
        planner.close()
        return True
