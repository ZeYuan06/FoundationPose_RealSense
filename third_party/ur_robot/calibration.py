"""
Hand-Eye Calibration: Camera-Base Transformation Estimation

This script solves the hand-eye calibration problem (AX=ZB) to find the transformation
between a robot's base frame and a camera frame. It solves for:
- X: tag to tool transformation (camera-to-gripper in standard notation)
- Z: camera to base transformation (derived from X)

The method uses OpenCV's hand-eye calibration for initial estimation and 
non-linear optimization to refine the transformation.

Required inputs:
- tool_poses_in_base: List of robot tool poses in base frame (at least 3 different poses)
- tag_poses_in_camera: List of calibration tag poses in camera frame (corresponding to tool poses)

Output:
- camera_H_base: Optimized transformation from camera to base frame
- tag_H_tool: Optimized transformation from tag to tool frame
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


# ============================================================================
# Helper Functions for Transformation Matrix Manipulation
# ============================================================================

def matrix_to_params(matrix):
    """
    Converts a 4x4 transformation matrix to a 6-DoF parameter vector.
    
    Args:
        matrix: 4x4 numpy array transformation matrix
        
    Returns:
        np.array: [tx, ty, tz, rx, ry, rz] where rotation is in axis-angle form
    """
    translation = matrix[:3, 3]
    rotation_vec = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return np.concatenate((translation, rotation_vec))


def params_to_matrix(params):
    """
    Converts a 6-DoF parameter vector back to a 4x4 transformation matrix.
    
    Args:
        params: np.array [tx, ty, tz, rx, ry, rz]
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    translation = params[:3]
    rotation_vec = params[3:]
    rotation_mat = R.from_rotvec(rotation_vec).as_matrix()
    
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation
    return transform_mat


def average_transforms(matrices_list):
    """
    Computes the average of multiple transformation matrices using quaternion averaging.
    
    This handles the special properties of SO(3) by averaging quaternions
    with sign disambiguation to avoid antipodal averaging issues.
    
    Args:
        matrices_list: List of 4x4 transformation matrices
        
    Returns:
        np.ndarray: 4x4 averaged transformation matrix
    """
    # Average translations
    translations = np.array([m[:3, 3] for m in matrices_list])
    avg_translation = np.mean(translations, axis=0)
    
    # Average rotations using quaternions
    quaternions = np.array([R.from_matrix(m[:3, :3]).as_quat() for m in matrices_list])
    
    # Ensure all quaternions are in the same hemisphere
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[0], quaternions[i]) < 0:
            quaternions[i] *= -1
    
    avg_quat = np.mean(quaternions, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    avg_rotation_mat = R.from_quat(avg_quat).as_matrix()
    
    # Construct averaged transformation matrix
    avg_matrix = np.eye(4)
    avg_matrix[:3, :3] = avg_rotation_mat
    avg_matrix[:3, 3] = avg_translation
    
    return avg_matrix


def xyz_rotvec_to_matrix(xyz_rotvec):
    """
    Converts a 6-DoF parameter vector [x, y, z, rx, ry, rz] to a 4x4 transformation matrix.
    
    Args:
        xyz_rotvec: np.array [x, y, z, rx, ry, rz] where rotation is in axis-angle form
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    translation = xyz_rotvec[:3]
    rotation_vec = xyz_rotvec[3:]
    rotation_mat = R.from_rotvec(rotation_vec).as_matrix()
    
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = translation
    return transform_mat


def invert_transform(matrix):
    """
    Inverts a 4x4 homogeneous transformation matrix.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        np.ndarray: Inverted 4x4 transformation matrix
    """
    inv_matrix = np.eye(4)
    R_inv = matrix[:3, :3].T
    inv_matrix[:3, :3] = R_inv
    inv_matrix[:3, 3] = -R_inv @ matrix[:3, 3]
    return inv_matrix


def matrix_to_xyz_rotvec(matrix):
    """
    Converts a 4x4 transformation matrix to [x, y, z, rx, ry, rz] format.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        np.array: [x, y, z, rx, ry, rz] where rotation is in axis-angle form (radians)
    """
    translation = matrix[:3, 3]
    rotation_vec = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return np.concatenate((translation, rotation_vec))


# ============================================================================
# Calibration Functions
# ============================================================================

def compute_initial_estimates(tool_poses_in_base, tag_poses_in_camera):
    """
    Computes initial estimates for both X (tag_H_tool) and Z (camera_H_base).
    
    This function uses a two-step approach:
    1. Use cv2.calibrateHandEye to solve: gripper2base * cam2gripper = target2cam
       which gives us cam2gripper (this is our tag_H_tool if tag is fixed to camera)
    2. Derive camera_H_base from the constraint: tool_base * tag_tool = camera_base * tag_camera
    
    Args:
        tool_poses_in_base: List of 4x4 transformation matrices (robot tool in base frame)
        tag_poses_in_camera: List of 4x4 transformation matrices (calibration tag in camera frame)
        
    Returns:
        tuple: (tag_H_tool_initial, camera_H_base_initial) - both as 4x4 matrices
    """
    # Prepare rotation and translation lists for OpenCV
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    for matrix in tool_poses_in_base:
        R_gripper2base.append(matrix[:3, :3])
        t_gripper2base.append(matrix[:3, 3].reshape(3, 1))
    
    for matrix in tag_poses_in_camera:
        R_target2cam.append(matrix[:3, :3])
        t_target2cam.append(matrix[:3, 3].reshape(3, 1))
    
    # Use OpenCV's hand-eye calibration
    R_tag2tool, t_tag2tool = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # Construct tag_H_tool (X)
    tag_H_tool_mat = np.eye(4)
    tag_H_tool_mat[:3, :3] = R_tag2tool
    tag_H_tool_mat[:3, 3] = t_tag2tool.flatten()
    tag_H_tool_initial = tag_H_tool_mat
    
    # Compute camera_H_base (Z) from first pose pair using: A * X = Z * B
    # Z = A * X * B^-1
    A = tool_poses_in_base[0]
    B = tag_poses_in_camera[0]
    X = tag_H_tool_initial
    
    camera_H_base_initial = A @ X @ invert_transform(B)
    
    return tag_H_tool_initial, camera_H_base_initial


def cost_function_dual(params, tool_poses_in_base, tag_poses_in_camera,
                       translation_weight=1.0, rotation_weight=1.0):
    """
    Computes residual vector for optimization of both X and Z.
    
    Parameterization: params = [x_params (6), z_params (6)]
    where each is [tx, ty, tz, rx, ry, rz] in axis-angle form.
    
    For each measurement pair, enforces the constraint: A_i * X = Z * B_i
    
    Args:
        params: np.array [x_tx, x_ty, x_tz, x_rx, x_ry, x_rz, z_tx, z_ty, z_tz, z_rx, z_ry, z_rz]
        tool_poses_in_base: List of 4x4 transformation matrices
        tag_poses_in_camera: List of 4x4 transformation matrices
        translation_weight: Weight for translation errors
        rotation_weight: Weight for rotation errors
        
    Returns:
        np.array: Residual vector (6 * N_measurements values)
    """
    # Split parameters into X and Z
    x_params = params[:6]
    z_params = params[6:]
    
    X = params_to_matrix(x_params)
    Z = params_to_matrix(z_params)
    
    residuals = []
    
    for A, B in zip(tool_poses_in_base, tag_poses_in_camera):
        # Compute error: E = (A * X) * (Z * B)^-1
        # If AX = ZB is satisfied, then E should be identity
        predicted = A @ X
        measured = Z @ B
        error_matrix = predicted @ invert_transform(measured)
        
        # Extract translation error (3 values)
        translation_error = error_matrix[:3, 3]
        
        # Extract rotation error in axis-angle form (3 values)
        rotation_error_vec = R.from_matrix(error_matrix[:3, :3]).as_rotvec()
        
        # Add weighted residuals
        residuals.extend(translation_weight * translation_error)
        residuals.extend(rotation_weight * rotation_error_vec)
    
    return np.array(residuals)


def optimize_hand_eye_calibration(tool_poses_in_base, tag_poses_in_camera, 
                                   initial_x=None, initial_z=None, verbose=True):
    """
    Optimizes both X (tag_H_tool) and Z (camera_H_base) transformations simultaneously.
    
    Args:
        tool_poses_in_base: List of 4x4 transformation matrices
        tag_poses_in_camera: List of 4x4 transformation matrices
        initial_x: 4x4 matrix for tag_H_tool (if None, will compute automatically)
        initial_z: 4x4 matrix for camera_H_base (if None, will compute automatically)
        verbose: Whether to print optimization progress
        
    Returns:
        tuple: (tag_H_tool, camera_H_base, result_object)
            - tag_H_tool: Optimized 4x4 transformation from tag to tool (X in AX=ZB)
            - camera_H_base: Optimized 4x4 transformation from camera to base (Z in AX=ZB)
            - result_object: OptimizeResult from scipy.optimize.least_squares
    """
    # Compute initial guesses if not provided
    if initial_x is None or initial_z is None:
        initial_x, initial_z = compute_initial_estimates(tool_poses_in_base, 
                                                         tag_poses_in_camera)
    
    # Combine initial parameters: [x_params, z_params]
    initial_params = np.concatenate([
        matrix_to_params(initial_x),
        matrix_to_params(initial_z)
    ])
    
    # Run optimization
    result = least_squares(
        fun=cost_function_dual,
        x0=initial_params,
        args=(tool_poses_in_base, tag_poses_in_camera),
        method='lm',  # Levenberg-Marquardt algorithm
        verbose=2 if verbose else 0
    )
    
    # Extract optimized matrices
    optimized_x = params_to_matrix(result.x[:6])
    optimized_z = params_to_matrix(result.x[6:])
    
    return optimized_x, optimized_z, result


# ============================================================================
# Main Calibration Script
# ============================================================================

def main():
    """
    Main function to perform hand-eye calibration.
    Fill in your measurement data in the section below.
    """
    
    # ========================================================================
    # TODO: Fill in your collected calibration data here
    # ========================================================================
    
    # Example format for creating transformation matrices from 4x4 numpy arrays:
    #
    # T_base_tool_1 = np.array([
    #     [r11, r12, r13, tx],
    #     [r21, r22, r23, ty],
    #     [r31, r32, r33, tz],
    #     [0,   0,   0,   1]
    # ], dtype=np.float64)
    #
    # tool_poses_in_base = [T_base_tool_1, T_base_tool_2, ...]
    # tag_poses_in_camera = [T_tag_cam_1, T_tag_cam_2, ...]

    # Example: Converting [x, y, z, rx, ry, rz] (axis-angle) to transformation matrix
    tool_pose_1 = xyz_rotvec_to_matrix(np.array([-0.70592, -0.34260, 0.16193, 1.329, 2.806, -0.553]))
    tool_pose_2 = xyz_rotvec_to_matrix(np.array([-0.79909, -0.31550, 0.16532, 1.070, 3.001, -0.584]))
    tool_pose_3 = xyz_rotvec_to_matrix(np.array([-0.67335, -0.51371, 0.16448, 1.665, 2.241, -0.844]))
    tool_pose_4 = xyz_rotvec_to_matrix(np.array([-0.90357, -0.34207, 0.16277, 0.714, 3.272, -0.823]))

    # Tag poses in camera frame (4x4 transformation matrices)
    tag_pose_1 = np.array([
        [-0.9687, 0.1054, 0.2248, 0.0315],
        [-0.1579, -0.9603, -0.2302, -0.0337],
        [0.1916, -0.2585, 0.9468, 0.3390],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ], dtype=np.float64)
    
    tag_pose_2 = np.array([
        [-0.9891, 0.1457, 0.0201, -0.0385],
        [-0.1465, -0.9647, -0.2189, -0.0302],
        [-0.0125, -0.2195, 0.9755, 0.3071],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ], dtype=np.float64)
    
    tag_pose_3 = np.array([
        [-0.7976, -0.0552, 0.6006, 0.1039],
        [0.0327, -0.9983, -0.0484, 0.0332],
        [0.6023, -0.0190, 0.7981, 0.2955],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ], dtype=np.float64)
    
    tag_pose_4 = np.array([
        [-0.9351, 0.2286, -0.2709, -0.0759],
        [-0.2105, -0.9730, -0.0944, -0.0002],
        [-0.2852, -0.0313, 0.9580, 0.2676],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ], dtype=np.float64)

    tool_poses_in_base = [tool_pose_1, tool_pose_2, tool_pose_3, tool_pose_4]
    tag_poses_in_camera = [tag_pose_1, tag_pose_2, tag_pose_3, tag_pose_4]

    # ========================================================================
    # Validation
    # ========================================================================
    
    if not tool_poses_in_base or not tag_poses_in_camera:
        raise ValueError("Please fill in the calibration data before running the script!")
    
    if len(tool_poses_in_base) != len(tag_poses_in_camera):
        raise ValueError("Number of tool poses and tag poses must match!")
    
    if len(tool_poses_in_base) < 3:
        raise ValueError("At least 3 pose pairs are required for calibration!")
    
    # ========================================================================
    # Step 1: Initial Estimation
    # ========================================================================
    
    print("=" * 70)
    print("HAND-EYE CALIBRATION: Camera-Base Transformation")
    print("=" * 70)
    print(f"\nNumber of measurement pairs: {len(tool_poses_in_base)}")
    # print("\nCalibration setup:")
    # print("  - tool_poses_in_base: Robot tool (gripper) poses in base coordinate frame")
    # print("  - tag_poses_in_camera: Calibration tag poses in camera coordinate frame")
    # print("  - Solving for:")
    # print("    * X = tag_H_tool (tag to tool transformation)")
    # print("    * Z = camera_H_base (camera to base transformation)")
    
    # print("\n--- Step 1: Computing Initial Estimates (using OpenCV) ---")
    tag_H_tool_initial, camera_H_base_initial = compute_initial_estimates(
        tool_poses_in_base, 
        tag_poses_in_camera
    )
    
    # print("Initial tag_H_tool estimate:")
    # print(tag_H_tool_initial.round(4))
    # print("\nInitial camera_H_base estimate:")
    # print(camera_H_base_initial.round(4))
    
    # ========================================================================
    # Step 2: Non-linear Optimization
    # ========================================================================
    
    # print("\n--- Step 2: Refining with Non-linear Optimization ---")
    tag_H_tool_optimized, camera_H_base_optimized, result = optimize_hand_eye_calibration(
        tool_poses_in_base,
        tag_poses_in_camera,
        initial_x=tag_H_tool_initial,
        initial_z=camera_H_base_initial,
        verbose=True
    )
    
    # ========================================================================
    # Display Results
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    
    print("\nOptimized tag_H_tool transformation (X in AX=ZB):")
    # print("4x4 Matrix:")
    # print(tag_H_tool_optimized.round(4))
    tag_H_tool_xyz_rotvec = matrix_to_xyz_rotvec(tag_H_tool_optimized)
    print("\n[x, y, z, rx, ry, rz] format (rotation in radians):")
    print(tag_H_tool_xyz_rotvec.round(4))
    
    print("\n" + "-" * 70)
    
    print("\nOptimized camera_H_base transformation (Z in AX=ZB):")
    # print("4x4 Matrix:")
    # print(camera_H_base_optimized.round(4))
    camera_H_base_xyz_rotvec = matrix_to_xyz_rotvec(camera_H_base_optimized)
    print("\n[x, y, z, rx, ry, rz] format (rotation in radians):")
    print(camera_H_base_xyz_rotvec.round(4))
    
    print(f"\nOptimization status: {result.message}")
    print(f"Final cost: {result.cost:.6f}")
    print(f"Number of iterations: {result.nfev}")
    
    print("\n" + "=" * 70)
    print("USAGE NOTES")
    print("=" * 70)
    print("camera_H_base: Use this to transform points from camera frame to robot base frame")
    print("tag_H_tool: Represents the fixed relationship between calibration tag and tool")
    print("\nNote: Rotation values (rx, ry, rz) are in axis-angle representation (radians)")
    
    return tag_H_tool_optimized, camera_H_base_optimized


if __name__ == "__main__":
    tag_H_tool, camera_H_base = main()
