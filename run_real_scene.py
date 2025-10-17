# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pyrealsense2 as rs
import numpy as np
from estimater import *
from datareader import *
import argparse
import subprocess
import os

class RealSenseReader:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))

        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()

        self.K = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                           [0, color_intrinsics.fy, color_intrinsics.ppy],
                           [0, 0, 1]])
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        """Capture a single frame of color and depth from the RealSense camera."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image
    
    def stop(self):
        self.pipeline.stop()

def get_mask_from_sam(image, output_path):
    """Generate mask using SAM in a different conda environment"""
    # Create temporary image path
    temp_image_path = output_path.replace('masks', 'temp').replace('.png', '_temp.png')
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
    cv2.imwrite(temp_image_path, image)
    
    script_path = os.path.join(os.path.dirname(__file__), "run_sam.sh")
    result = subprocess.run([script_path, temp_image_path, output_path], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running SAM script:", result.stderr)
        return None
    
    # Clean up temp file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    # Read the generated mask
    if os.path.exists(output_path):
        mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        return mask > 0  # Binary
    return None

def save_realsense_data(output_dir, frame_id, color_image, depth_image, camera_matrix):
    """Save RealSense data in the expected format"""
    # Create directories
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Save files with proper naming convention
    frame_name = f"{frame_id:06d}.png"
    
    # Save color image
    rgb_path = os.path.join(rgb_dir, frame_name)
    cv2.imwrite(rgb_path, color_image)
    
    # Save depth image
    depth_path = os.path.join(depth_dir, frame_name)
    cv2.imwrite(depth_path, depth_image)

    # Save camera intrinsics
    cam_k_path = os.path.join(output_dir, 'cam_K.txt')
    np.savetxt(cam_k_path, camera_matrix)
    
    return rgb_path, depth_path, os.path.join(masks_dir, frame_name)

def load_mesh_from_file(mesh_file):
    """Load mesh from file, handling both .obj and .glb formats"""
    mesh_obj = trimesh.load(mesh_file)
    
    if isinstance(mesh_obj, trimesh.Scene):
        # For .glb files, extract the mesh from the scene
        if len(mesh_obj.geometry) == 0:
            raise ValueError(f"No geometry found in {mesh_file}")
        
        # Get the first mesh from the scene
        mesh_name = list(mesh_obj.geometry.keys())[0]
        mesh = mesh_obj.geometry[mesh_name]
        
        # Apply any transforms if present
        transform = mesh_obj.graph.get(mesh_name)[0]
        if transform is not None:
            mesh.apply_transform(transform)
        
        # Convert GLB to OBJ format to avoid PBR material issues
        if mesh_file.lower().endswith('.glb'):
            obj_file = mesh_file.replace('.glb', '_converted.obj')
            print(f"Converting GLB to OBJ: {obj_file}")
            
            # Remove problematic visual properties that cause PBR issues
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                # Create a simple white material or extract base color
                try:
                    if hasattr(mesh.visual.material, 'baseColorFactor'):
                        # Use base color from PBR material
                        base_color = mesh.visual.material.baseColorFactor[:3]  # RGB only
                        mesh.visual = trimesh.visual.ColorVisuals(
                            mesh=mesh, 
                            face_colors=np.full((len(mesh.faces), 4), [*base_color, 255], dtype=np.uint8)
                        )
                    else:
                        # Default to white
                        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
                except:
                    # If anything fails, just remove visual properties
                    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
            
            # Export as OBJ
            try:
                mesh.export(obj_file)
                print(f"Successfully converted to: {obj_file}")
            except Exception as e:
                print(f"Warning: Could not export OBJ file: {e}")
                print("Continuing with original mesh...")
            
    elif isinstance(mesh_obj, trimesh.Trimesh):
        # For .obj files, use directly
        mesh = mesh_obj
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh_obj)}")
    
    # Ensure the mesh has vertex normals
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.vertex_normals = mesh.vertex_normals  # This will compute them if needed
    
    # Final check: if mesh still has problematic visual properties, remove them
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        if hasattr(mesh.visual.material, 'baseColorTexture') or not hasattr(mesh.visual.material, 'image'):
            print("Removing problematic visual properties...")
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    mesh.apply_scale(0.001)
    
    return mesh

def run_pose_estimation_on_captured_data(output_dir, mesh_file, est_refine_iter=5, track_refine_iter=2, debug=1):
    """Run pose estimation on the captured RealSense data"""
    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = os.path.join(code_dir, 'debug')
    
    # Clean debug directory
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
    
    # Load mesh
    mesh = load_mesh_from_file(mesh_file)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    # Setup pose estimator
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    
    logging.info("estimator initialization done")
    
    # Create data reader for captured data
    reader = YcbineoatReader(video_dir=output_dir, shorter_side=None, zfar=np.inf)
    
    poses = []
    
    for i in range(len(reader.color_files)):
        logging.info(f'Processing frame {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        
        if i == 0:
            # First frame: registration
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
            
            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            # Subsequent frames: tracking
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
        
        center_pose = pose @ np.linalg.inv(to_origin)
        poses.append(center_pose)

        # Save pose
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', center_pose.reshape(4,4))
        
        if debug >= 1:
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('Pose Estimation Result', vis[...,::-1])
            cv2.waitKey(1)
        
        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    return poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/realsense/mesh/apple.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--mode', type=str, choices=['capture', 'estimate', 'both'], default='both',
                       help='Mode: capture data, estimate poses, or both')
    parser.add_argument('--output_dir', type=str, default=f'{code_dir}/demo_data/realsense/',
                       help='Directory to save/load data')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    if args.mode in ['capture', 'both']:
        # Data capture phase
        print("=== DATA CAPTURE PHASE ===")
        reader = RealSenseReader()
        frame_count = 0
        
        print("RealSense initialized. Controls:")
        print("- Press 's' to save current frame and generate mask")
        print("- Press 'c' to finish capture and start pose estimation")
        print("- Press 'q' to quit")
        print(f"Data will be saved to: {args.output_dir}")

        try:
            while True:
                color_image, depth_image = reader.get_frame()

                if color_image is None or depth_image is None:
                    print("Failed to get frame.")
                    continue

                # Convert depth for visualization
                depth_for_viz = depth_image.astype(np.float32)
                # Normalize to 0-255 range for better visualization
                depth_min, depth_max = 0, 4000
                depth_normalized = np.clip((depth_for_viz - depth_min) / (depth_max - depth_min) * 255, 0, 255)
                depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # Display the images
                cv2.imshow("RealSense Color", color_image)
                cv2.imshow("RealSense Depth", depth_colormap)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and frame_count > 0:
                    print("Finishing capture and starting pose estimation...")
                    break
                elif key == ord('s'):
                    print(f"Saving frame {frame_count:06d}...")
                    
                    # Save RealSense data
                    rgb_path, depth_path, mask_path = save_realsense_data(
                        args.output_dir, frame_count, color_image, depth_image, reader.K
                    )
                    print(f"Saved RGB: {rgb_path}")
                    print(f"Saved Depth: {depth_path}")
                    
                    # Generate mask with SAM
                    print("Generating mask with SAM...")
                    mask = get_mask_from_sam(color_image, mask_path)
                    if mask is not None:
                        print(f"Saved Mask: {mask_path}")
                        # Display the mask
                        cv2.imshow("Generated Mask", mask.astype(np.uint8) * 255)
                    else:
                        print("Failed to generate mask")
                    
                    frame_count += 1

        finally:
            print("Stopping camera...")
            reader.stop()
            cv2.destroyAllWindows()
        
        if frame_count == 0:
            print("No frames captured. Exiting.")
            exit()
        
        print(f"\nCaptured {frame_count} frames to {args.output_dir}")

    if args.mode in ['estimate', 'both']:
        # Check if data exists
        if not os.path.exists(os.path.join(args.output_dir, 'rgb')):
            print(f"No captured data found in {args.output_dir}. Please capture data first.")
            exit()
        
        # Pose estimation phase
        print("\n=== POSE ESTIMATION PHASE ===")
        print("Running pose estimation on captured data...")
        
        poses = run_pose_estimation_on_captured_data(
            args.output_dir, 
            args.mesh_file,
            args.est_refine_iter,
            args.track_refine_iter,
            args.debug
        )
        
        print(f"\nPose estimation completed! {len(poses)} poses estimated.")
        print("Results saved in debug/ directory")
        
        # Keep visualization window open
        if args.debug >= 1:
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
