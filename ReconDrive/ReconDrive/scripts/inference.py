#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
Scene-based inference script for ReconDrive
This script demonstrates inference using scene-by-scene iteration
"""

import yaml
import argparse
import os
import sys
import torch
import json
from pathlib import Path
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import (
    quat_scale_to_covar_preci,
    world_to_cam,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
    spherical_harmonics,
)
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))  # Add models directory for vggt imports

from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
from dataset.vggt4dgs_scene_dataset import custom_collate_fn
from models.recondrive_model import ReconDrive_LITModelModule


class SceneSampleDataset(Dataset):
    """Dataset wrapper that supports both pre-loaded samples and lazy loading"""
    
    def __init__(self, samples_or_indices, dataset=None, scene_idx=None):
        if dataset is not None and scene_idx is not None:
            # Lazy loading mode: samples_or_indices is list of sample indices
            self.lazy_mode = True
            self.sample_indices = samples_or_indices
            self.dataset = dataset
            self.scene_idx = scene_idx
        else:
            # Pre-loaded mode: samples_or_indices is list of actual samples
            self.lazy_mode = False
            self.samples = samples_or_indices
    
    def __len__(self):
        if self.lazy_mode:
            return len(self.sample_indices) 
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if self.lazy_mode:
            # Load sample on-demand
            sample_idx = self.sample_indices[idx] 
            return self.dataset.__getitem__(sample_idx,self.scene_idx)
        else:
            return self.samples[idx]



def load_model_from_checkpoint(checkpoint_path, model_cfg, device):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Ensure batch_size is in model config
    if 'batch_size' not in model_cfg:
        model_cfg['batch_size'] = 1  # Set default batch_size for inference
    
    # Initialize model
    model = ReconDrive_LITModelModule(
        cfg=model_cfg,
        save_dir='./temp_log',
        logger=None
    )

    model.load_pretrained_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def run_inference(model_cfg=None, model=None, checkpoint_path=None,
                  scene_dataloader=None, device='cuda:0',
                  save_results=True, output_dir=None, novel_distances=[1.0, 2.0],
                  eval_resolution='280x518', bev_x_range=50.0, bev_y_range=25.0,
                  bev_resolution=0.2, output_gs=False):
    """
    Scene-based inference function for single GPU

    Args:
        model_cfg: Model configuration (when model=None)
        model: Pre-loaded model (optional)
        checkpoint_path: Path to model checkpoint
        scene_dataloader: Scene data loader
        device: Device string (e.g., 'cuda:0')
        save_results: Whether to save results
        output_dir: Output directory
        novel_distances: List of distances for novel view generation
        eval_resolution: Resolution mode - 'original' or 'upsampled'
    """
    print(f"\nStarting single-GPU scene-based inference on {device}")
    print(f"Number of scenes to process: {len(scene_dataloader)}")

    # Load model if not provided
    if model is None:
        if model_cfg is None or checkpoint_path is None:
            raise ValueError("model_cfg and checkpoint_path required when model is None")
        model = load_model_from_checkpoint(checkpoint_path, model_cfg, device)

    return _run_single_gpu_inference(model, scene_dataloader, device, save_results, output_dir, novel_distances, eval_resolution,
                                     bev_x_range, bev_y_range, bev_resolution, output_gs)


def save_rendered_image(tensor_img, save_path, upsample_to=None):
    """Save a tensor image to file with optional upsampling"""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)

    if upsample_to is not None:
        target_height, target_width = upsample_to
        device = tensor_img.device
        if tensor_img.device.type == 'cpu':
            tensor_img = tensor_img.cuda()
        
        tensor_img = tensor_img.unsqueeze(0)
        tensor_img = F.interpolate(tensor_img, size=(target_height, target_width), 
                                 mode='bilinear', align_corners=False)
        tensor_img = tensor_img.squeeze(0)
        
        if device.type == 'cpu':
            tensor_img = tensor_img.cpu()
    
    img_np = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    Image.fromarray(img_np).save(save_path)


def create_lateral_translation_matrices(translation_distances=[1.0, 2.0]):
    """Create transformation matrices for lateral (left/right) ego vehicle translation"""
    transforms = {}
    
    for dist in translation_distances:
        left_transform = torch.eye(4)
        left_transform[1, 3] = dist  # Negative Y for left
        transforms[f'left_{dist}m'] = left_transform
        
        right_transform = torch.eye(4)
        right_transform[1, 3] = -dist   # Positive Y for right
        transforms[f'right_{dist}m'] = right_transform
    
    return transforms


def render_novel_views(model, recontrast_data, render_data, device, scene_name, sample_idx, save_dir, actual_sample_idx, translation_distances=[1.0, 2.0], eval_resolution='280x518', novel_render_frames=[]):
    """Render novel views with lateral ego translation"""
    # Get transformation matrices for translation
    translation_transforms = create_lateral_translation_matrices(translation_distances)
    
    saved_paths = []
    
    xyz_i = recontrast_data['xyz'][sample_idx:sample_idx+1]  # Keep batch dimension
    rot_i = recontrast_data['rot_maps'][sample_idx:sample_idx+1]
    scale_i = recontrast_data['scale_maps'][sample_idx:sample_idx+1]
    opacity_i = recontrast_data['opacity_maps'][sample_idx:sample_idx+1]
    sh_i = recontrast_data['sh_maps'][sample_idx:sample_idx+1]
    
    # Get camera parameters
    frame_id = 0  # Use current frame

    num_cams = getattr(model, 'num_cams', 6)
    model_width = getattr(model, 'width', 518)
    model_height = getattr(model, 'height', 280)
    for transform_name, transform_matrix in translation_transforms.items():
        transform_matrix = transform_matrix.to(device)
        for frame_id in novel_render_frames:
            for cam_id in range(num_cams):
                # Get original camera extrinsics and intrinsics
                original_e2c_extr = render_data[('e2c_extr', frame_id, cam_id)][sample_idx:sample_idx+1]
                K_i = render_data[('K', frame_id, cam_id)][sample_idx:sample_idx+1, :3, :3]
                
                # Apply lateral translation to camera pose
                # Transform ego to camera: new_e2c = e2c @ inv(transform)
                novel_e2c_extr = torch.matmul(original_e2c_extr, torch.linalg.inv(transform_matrix.unsqueeze(0)))
                
                # Render with new camera pose
                render_colors_i, render_alphas_i, meta_i = rasterization(
                    xyz_i.squeeze(0),      # [N, 3]
                    rot_i.squeeze(0),      # [N, 4]
                    scale_i.squeeze(0),    # [N, 3]
                    opacity_i.squeeze(0).squeeze(-1),  # [N]
                    sh_i.squeeze(0),       # [N, K, 3]
                    velocities=None,
                    viewmats=novel_e2c_extr,  # [1, 4, 4]
                    Ks=K_i,                   # [1, 3, 3]
                    width=model_width,
                    height=model_height,
                    sh_degree=getattr(model, 'sh_degree', 3),
                    render_mode="RGB",
                )
                
                # Extract RGB and convert to proper format
                render_rgb = render_colors_i[..., :3].permute(0, 3, 1, 2)[0]  # [C, H, W]
                
                # Save the novel view
                global_sample_idx =  actual_sample_idx + frame_id
                save_path = os.path.join(save_dir, scene_name, 
                                        f'sample_{global_sample_idx:04d}', transform_name, f'{eval_resolution}_cam_{cam_id}.png')
            
                resize_height, resize_width = eval_resolution.split('x')
                resize_height = int(resize_height)
                resize_width = int(resize_width)
                if resize_height !=model_height or resize_width != model_width:
                    save_rendered_image(render_rgb, save_path, upsample_to=(resize_height, resize_width))
                else:  # eval_resolution == 'original'
                    save_rendered_image(render_rgb, save_path)
        
    return saved_paths

def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device)) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data

def _extract_first_value(value, default=None):
    """Extract first value from list/tensor or return as-is"""
    if isinstance(value, list):
        return value[0] if value else default
    elif isinstance(value, torch.Tensor):
        if value.numel() > 0:
            return value[0].item() if value.dim() > 0 else value.item()
        return default
    return value

def _extract_scene_idx(scene_batch, default_idx=0):
    """Extract scene_idx from batch, checking multiple locations"""
    for key in ['target_frames', 'context_frames']:
        if key in scene_batch and 'scene_idx' in scene_batch[key]:
            return _extract_first_value(scene_batch[key]['scene_idx'], default_idx)
    return default_idx

def _process_scene_batch(model, scene_batch, device, gpu_id=0, save_renders=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518', batch_idx=0,
                         bev_x_range=50.0, bev_y_range=25.0, bev_resolution=0.2, output_gs=False):
    """Process a single scene batch and return results"""
    scene_start_time = time.time()

    scene_name = scene_batch['scene_name']
    scene_token = scene_batch['scene_token']
    
    # Initialize original_sample_indices to track the actual frame indices
    original_sample_indices = None
    frame_skip = 6

    # Support both lazy loading and pre-loaded modes
    if 'samples' in scene_batch:
        scene_samples = scene_batch['samples']
        scene_length = len(scene_samples)
        
        if frame_skip is not None and frame_skip > 1:
            # Calculate which indices to keep: 0, frame_skip, 2*frame_skip, ...
            original_sample_indices = list(range(0, scene_length, frame_skip))
            filtered_samples = [scene_samples[i] for i in original_sample_indices]
            scene_dataset = SceneSampleDataset(filtered_samples)
            actual_samples = len(filtered_samples)
            
            print(f"GPU {gpu_id}: Using frame skip of {frame_skip}, processing {actual_samples} out of {scene_length} samples")
        else:
            scene_dataset = SceneSampleDataset(scene_samples)
            actual_samples = scene_length
            original_sample_indices = list(range(scene_length))
    else:
        scene_length = scene_batch['scene_length']
        all_indices = scene_batch['sample_indices']
        
        if frame_skip is not None and frame_skip > 1:
            # Calculate which indices to keep: 0, frame_skip, 2*frame_skip, ...
            positions_to_keep = list(range(0, len(all_indices), frame_skip))
            original_sample_indices = [all_indices[pos] for pos in positions_to_keep]
            
            print(f"GPU {gpu_id}: Using frame skip of {frame_skip}, processing {len(original_sample_indices)} out of {scene_length} samples")
            
            scene_dataset = SceneSampleDataset(
                original_sample_indices,
                dataset=scene_batch['dataset'], 
                scene_idx=scene_batch['scene_idx']
            )
            actual_samples = len(original_sample_indices)
        else:
            original_sample_indices = all_indices
            scene_dataset = SceneSampleDataset(
                all_indices,
                dataset=scene_batch['dataset'], 
                scene_idx=scene_batch['scene_idx']
            )
            actual_samples = len(all_indices)
    
    print(f"GPU {gpu_id}: Processing Scene: {scene_name} ({actual_samples} samples)")
    
    # Create DataLoader
    scene_loader = DataLoader(
        scene_dataset, batch_size=1, shuffle=False,
        pin_memory=False, num_workers=0, drop_last=False,
        collate_fn=custom_collate_fn
    )

    scene_psnr_list, scene_ssim_list, scene_lpips_list = [], [], []
    lidar_depth_l2_list, lidar_depth_median_l2_list = [], []
    lidar_delta_1_list, lidar_delta_2_list, lidar_delta_3_list = [], [], []
    lidar_intensity_rmse_list, lidar_ray_drop_acc_list = [], []
    lidar_chamfer_dist_list = []
    batch_count = 0

    for batch_data in scene_loader:
        batch_count += 1
        
        # Get the actual sample index from the original data
        if original_sample_indices is not None:
            if batch_count - 1 < len(original_sample_indices):
                actual_sample_idx = original_sample_indices[batch_count - 1]
            else:
                actual_sample_idx = batch_count - 1
        else:
            actual_sample_idx = batch_count - 1

        num_cams = getattr(model, 'num_cams', 6)

        batch_data = to_device(batch_data, device)

        # Run prediction
        output = model.predict_step(batch_data, batch_idx)

        model_width = getattr(model, 'width', 518)
        model_height = getattr(model, 'height', 280)

        # --- LiDAR Rendering ---
        lidar_out = None
        lidar_gt = None
        if isinstance(output, tuple):
            batch_recontrast_data = output[0]
            lidar_out = model.render_lidar(batch_recontrast_data, batch_data)
            if lidar_out is not None:
                lidar_gt = batch_data.get('lidar', None)
        # -----------------------

        if isinstance(output, tuple):
            batch_recontrast_data, batch_render_data, batch_splating_data = output
            
            # Calculate metrics for this batch
            batch_psnr, batch_ssim, batch_lpips = [], [], []

            # Separate metrics for reconstruction and novel view modes
            recon_psnr, recon_ssim, recon_lpips = [], [], []
            novel_psnr, novel_ssim, novel_lpips = [], [], []

            # Use the batch index as the global sample index


            # Determine which frames are available from the returned data
            available_frames = set()
            for key in batch_splating_data.keys():
                if isinstance(key, tuple) and key[0] == 'gaussian_color':
                    available_frames.add(key[1])  # frame_id is at index 1
            available_frames = sorted(list(available_frames))

            print(f"GPU {gpu_id}: Available frames for evaluation: {available_frames}")

            # === Mode 1: Scene Reconstruction (frame 0) ===
            frame_id = 0
            if frame_id in [0]:
                for cam_id in range(num_cams):
                    pred_key = ('gaussian_color', frame_id, cam_id)
                    gt_key = ('groudtruth', frame_id, cam_id)
                    if pred_key in batch_splating_data and gt_key in batch_splating_data:
                        pred = batch_splating_data[pred_key][0:1]
                        gt = batch_splating_data[gt_key][0:1]

                        resize_height, resize_width = eval_resolution.split('x')
                        resize_height = int(resize_height)
                        resize_width = int(resize_width)
                        if (resize_height == model_height) and (resize_width == model_width):
                            pred_eval = pred.clamp(0, 1)
                            gt_eval = gt.clamp(0, 1)
                        else:
                            if ('color_org', frame_id) in scene_batch:
                                gt_original = scene_batch[('color_org', frame_id)][:, cam_id, ...][0:1]
                                gt_original = gt_original.clamp(0, 1).to(pred.device)
                                gt_eval = F.interpolate(gt_original, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            else:
                                gt_eval = F.interpolate(gt, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                                gt_eval = gt_eval.clamp(0, 1)
                            pred_eval = F.interpolate(pred, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            pred_eval = pred_eval.clamp(0, 1)

                        psnr_val = model.compute_psnr(gt_eval, pred_eval).mean().item()
                        ssim_val = model.compute_ssim(gt_eval, pred_eval).mean().item()
                        lpips_val = model.compute_lpips(gt_eval, pred_eval).mean().item()

                        recon_psnr.append(psnr_val)
                        recon_ssim.append(ssim_val)
                        recon_lpips.append(lpips_val)

                        if output_dir:
                            global_sample_idx = actual_sample_idx + frame_id
                            frame_dir = 'gt_views'
                            pred_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_pred.png')
                            gt_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_gt.png')
                            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                            save_rendered_image(pred_eval.squeeze(0), pred_save_path)
                            save_rendered_image(gt_eval.squeeze(0), gt_save_path)

            # === Mode 2: Novel View Synthesis (middle frames) ===
            novel_frames = [f for f in available_frames if f != 0]
            for frame_id in novel_frames:
                for cam_id in range(num_cams):
                    pred_key = ('gaussian_color', frame_id, cam_id)
                    gt_key = ('groudtruth', frame_id, cam_id)
                    if pred_key in batch_splating_data and gt_key in batch_splating_data:
                        pred = batch_splating_data[pred_key][0:1]
                        gt = batch_splating_data[gt_key][0:1]

                        resize_height, resize_width = eval_resolution.split('x')
                        resize_height = int(resize_height)
                        resize_width = int(resize_width)
                        if (resize_height == model_height) and (resize_width == model_width):
                            # Original mode: Use original model resolution (280x518)
                            if frame_id == 0 and cam_id == 0:
                                print(f"GPU {gpu_id}: Original mode - Using model resolution: pred={pred.shape}, gt={gt.shape}")
                                print(f"GPU {gpu_id}: Original mode - pred range: [{pred.min():.3f}, {pred.max():.3f}], gt range: [{gt.min():.3f}, {gt.max():.3f}]")
                            
                            # Ensure both pred and gt are in [0,1] range
                            pred_eval = pred.clamp(0, 1)
                            gt_eval = gt.clamp(0, 1)
                        else:
                            if ('color_org', frame_id) in scene_batch:
                                # Use original high-resolution GT and upsample to 900x1600
                                gt_original = scene_batch[('color_org', frame_id)][:, cam_id, ...][0:1]
                                if frame_id == 0 and cam_id == 0:
                                    print(f"GPU {gpu_id}: Upsampled mode - Original GT shape: {gt_original.shape}")
                                    print(f"GPU {gpu_id}: Upsampled mode - Original GT range: [{gt_original.min():.3f}, {gt_original.max():.3f}]")
                                
                                # Ensure GT is in [0,1] range and on correct device
                                gt_original = gt_original.clamp(0, 1).to(pred.device)
                                gt_eval = F.interpolate(gt_original, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            else:
                                # Fallback: use downsampled GT if original not available
                                if frame_id == 0 and cam_id == 0:
                                    print(f"GPU {gpu_id}: Upsampled mode - Warning: Using downsampled GT: {gt.shape}")
                                gt_eval = F.interpolate(gt, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                                gt_eval = gt_eval.clamp(0, 1)
                            
                            # Upsample predicted image to 900x1600 and ensure in [0,1] range
                            pred_eval = F.interpolate(pred, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            pred_eval = pred_eval.clamp(0, 1)

                        # Calculate novel view metrics
                        psnr_val = model.compute_psnr(gt_eval, pred_eval).mean().item()
                        ssim_val = model.compute_ssim(gt_eval, pred_eval).mean().item()
                        lpips_val = model.compute_lpips(gt_eval, pred_eval).mean().item()

                        novel_psnr.append(psnr_val)
                        novel_ssim.append(ssim_val)
                        novel_lpips.append(lpips_val)

                        # Save novel view images
                        if output_dir:
                            global_sample_idx = actual_sample_idx + frame_id
                            frame_dir = f'gt_views'
                            pred_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_pred.png')
                            gt_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_gt.png')
                            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                            save_rendered_image(pred_eval.squeeze(0), pred_save_path)
                            save_rendered_image(gt_eval.squeeze(0), gt_save_path)

            # Print separate metrics for both modes
            if recon_psnr:
                print(f"GPU {gpu_id}: Recon metrics - PSNR: {np.mean(recon_psnr):.3f}, SSIM: {np.mean(recon_ssim):.3f}, LPIPS: {np.mean(recon_lpips):.3f}")
            if novel_psnr:
                print(f"GPU {gpu_id}: Novel metrics - PSNR: {np.mean(novel_psnr):.3f}, SSIM: {np.mean(novel_ssim):.3f}, LPIPS: {np.mean(novel_lpips):.3f}")

            # Generate and save novel views for all samples
            if save_renders and output_dir:
                # Define which frames to render novel views for (use available frames by default)
                novel_render_frames = available_frames  # Use only available frames
                novel_view_paths = render_novel_views(
                    model, batch_recontrast_data, batch_render_data,
                    device, scene_name, 0, output_dir, actual_sample_idx, novel_distances, eval_resolution, novel_render_frames
                )
                print(f"GPU {gpu_id}: Saved novel views for sample {actual_sample_idx}: {len(novel_view_paths)} images")

            # --- Save LiDAR results ---
            if lidar_out is not None and output_dir:
                global_sample_idx = actual_sample_idx
                lidar_dir = os.path.join(output_dir, scene_name, f'sample_{global_sample_idx:04d}', 'lidar')
                os.makedirs(lidar_dir, exist_ok=True)

                def _save_lidar_map(arr, path):
                    """Squeeze to 2D and save as PNG"""
                    arr_2d = np.ascontiguousarray(arr.squeeze())
                    if arr_2d.ndim == 1:
                        arr_2d = arr_2d.reshape(1, -1)
                    Image.fromarray(arr_2d).save(path)

                # Predicted depth [B, H, W, 1]
                pred_depth = lidar_out['depth'][0]  # [H, W, 1]
                pred_depth_np = pred_depth.detach().cpu().numpy()
                depth_mm = np.clip(pred_depth_np * 1000, 0, 65535).astype(np.uint16)
                _save_lidar_map(depth_mm, os.path.join(lidar_dir, 'pred_depth.png'))

                # Predicted intensity [B, H, W, 1]
                pred_intensity = lidar_out['intensity'][0]  # [H, W, 1]
                pred_intensity_np = (pred_intensity.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                _save_lidar_map(pred_intensity_np, os.path.join(lidar_dir, 'pred_intensity.png'))

                # Predicted ray_drop probability [B, H, W, 1]
                pred_ray_drop = lidar_out['ray_drop_logits'][0].sigmoid()  # [H, W, 1]
                pred_ray_drop_np = (pred_ray_drop.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                _save_lidar_map(pred_ray_drop_np, os.path.join(lidar_dir, 'pred_ray_drop.png'))

                # Ground truth (if available)
                if lidar_gt is not None:
                    gt_depth = lidar_gt['gt_depth'][0]
                    gt_depth_np = gt_depth.detach().cpu().numpy()
                    gt_depth_mm = np.clip(gt_depth_np * 1000, 0, 65535).astype(np.uint16)
                    _save_lidar_map(gt_depth_mm, os.path.join(lidar_dir, 'gt_depth.png'))

                    gt_intensity = lidar_gt['gt_intensity'][0]
                    gt_intensity_np = (gt_intensity.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    _save_lidar_map(gt_intensity_np, os.path.join(lidar_dir, 'gt_intensity.png'))

                    gt_ray_drop = lidar_gt['gt_ray_drop'][0]
                    gt_ray_drop_np = (gt_ray_drop.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    _save_lidar_map(gt_ray_drop_np, os.path.join(lidar_dir, 'gt_ray_drop.png'))

                    # Compute LiDAR metrics
                    valid = gt_depth > 0
                    
                    # Depth L2 (MSE)
                    depth_sq = (pred_depth[valid] - gt_depth[valid]) ** 2
                    depth_l2 = depth_sq.mean().item() if valid.any() else 0.0
                    
                    # Median Depth L2
                    depth_median_l2 = depth_sq.median().item() if valid.any() else 0.0

                    # δ accuracy (depth ratio thresholds)
                    delta_1 = 0.0
                    delta_2 = 0.0
                    delta_3 = 0.0
                    if valid.any():
                        pred_d_valid = pred_depth[valid]
                        gt_d_valid = gt_depth[valid]
                        depth_ratio = torch.max(pred_d_valid / gt_d_valid, gt_d_valid / pred_d_valid)
                        delta_1 = (depth_ratio < 1.25).float().mean().item()
                        delta_2 = (depth_ratio < 1.25**2).float().mean().item()
                        delta_3 = (depth_ratio < 1.25**3).float().mean().item()

                    # Intensity RMSE
                    intensity_sq = ((pred_intensity[valid] - gt_intensity[valid]) ** 2)
                    intensity_rmse = intensity_sq.mean().sqrt().item() if valid.any() else 0.0
                    
                    # Ray Drop Acc
                    ray_drop_acc = ((pred_ray_drop > 0.5) == (gt_ray_drop > 0.5)).float().mean().item()

                    # Chamfer Distance (3D point cloud, subsampled for memory)
                    chamfer_dist = 0.0
                    if valid.any():
                        raster_pts = lidar_gt['raster_pts'][0]  # [H, W, 4]
                        az_rad = torch.deg2rad(raster_pts[..., 0:1])    # [H, W, 1]
                        el_rad = torch.deg2rad(raster_pts[..., 1:2])    # [H, W, 1]
                        cos_el = torch.cos(el_rad)
                        # GT & Pred 3D points
                        gt_d = gt_depth.expand_as(az_rad)
                        pred_d = pred_depth.expand_as(az_rad)
                        x_gt = gt_d * cos_el * torch.cos(az_rad)
                        y_gt = gt_d * cos_el * torch.sin(az_rad)
                        z_gt = gt_d * torch.sin(el_rad)
                        x_pred = pred_d * cos_el * torch.cos(az_rad)
                        y_pred = pred_d * cos_el * torch.sin(az_rad)
                        z_pred = pred_d * torch.sin(el_rad)
                        pts_gt = torch.stack([x_gt, y_gt, z_gt], dim=-1)   # [H, W, 3]
                        pts_pred = torch.stack([x_pred, y_pred, z_pred], dim=-1)
                        pts_gt_v = pts_gt[valid.squeeze(-1)]    # [N, 3]
                        pts_pred_v = pts_pred[valid.squeeze(-1)] # [N, 3]
                        N = pts_gt_v.shape[0]
                        # Subsample to at most 2048 points to keep O(N^2) cdist tractable
                        max_pts = 2048
                        if N > max_pts:
                            idx = torch.randperm(N, device=pts_gt_v.device)[:max_pts]
                            pts_gt_v = pts_gt_v[idx]
                            pts_pred_v = pts_pred_v[idx]
                        dist_g2p = torch.cdist(pts_gt_v, pts_pred_v).min(dim=1).values.mean().item()
                        dist_p2g = torch.cdist(pts_pred_v, pts_gt_v).min(dim=1).values.mean().item()
                        chamfer_dist = dist_g2p + dist_p2g

                    print(f"GPU {gpu_id}: LiDAR metrics - DepthL2: {depth_l2:.4f}, MedDepthL2: {depth_median_l2:.4f}, "
                          f"δ1: {delta_1:.4f}, δ2: {delta_2:.4f}, δ3: {delta_3:.4f}, "
                          f"IntRMSE: {intensity_rmse:.4f}, RayDropAcc: {ray_drop_acc:.4f}, ChamferDist: {chamfer_dist:.4f}")

                    # Accumulate
                    lidar_depth_l2_list.append(depth_l2)
                    lidar_depth_median_l2_list.append(depth_median_l2)
                    lidar_delta_1_list.append(delta_1)
                    lidar_delta_2_list.append(delta_2)
                    lidar_delta_3_list.append(delta_3)
                    lidar_intensity_rmse_list.append(intensity_rmse)
                    lidar_ray_drop_acc_list.append(ray_drop_acc)
                    lidar_chamfer_dist_list.append(chamfer_dist)
            # -------------------------

            # --- Save LiDAR-camera overlays ---
            if lidar_out is not None and lidar_gt is not None and output_dir:
                save_lidar_cam_overlays(
                    lidar_gt, lidar_out, batch_render_data, batch_splating_data,
                    scene_name, actual_sample_idx, output_dir, num_cams=num_cams,
                )
            # -------------------------

            # --- Save BEV (bird's eye view) outputs ---
            if output_dir:
                try:
                    bev_paths = render_bev_views(
                        model, batch_recontrast_data, device, scene_name, actual_sample_idx,
                        output_dir, bev_x_range, bev_y_range, bev_resolution,
                    )
                    print(f"GPU {gpu_id}: Saved BEV views for sample {actual_sample_idx}: {len(bev_paths)} images")

                    if lidar_out is not None and lidar_gt is not None:
                        lidar_bev_paths = save_lidar_bev(
                            lidar_gt, lidar_out, scene_name, actual_sample_idx,
                            output_dir, bev_x_range, bev_y_range, bev_resolution,
                        )
                        print(f"GPU {gpu_id}: Saved LiDAR BEV maps for sample {actual_sample_idx}: {len(lidar_bev_paths)} images")
                except Exception as e:
                    print(f"GPU {gpu_id}: [WARN] BEV output failed for sample {actual_sample_idx}: {e}")
            # -------------------------

            # --- Save LiDAR point clouds (PLY) ---
            if lidar_out is not None and lidar_gt is not None and output_dir:
                try:
                    ply_paths = save_lidar_ply(
                        lidar_gt, lidar_out, scene_name, actual_sample_idx, output_dir,
                    )
                    ply_names = [os.path.basename(p) for p in ply_paths]
                    print(f"GPU {gpu_id}: Saved LiDAR point clouds for sample {actual_sample_idx}: {ply_names}")
                except Exception as e:
                    print(f"GPU {gpu_id}: [WARN] LiDAR PLY output failed for sample {actual_sample_idx}: {e}")
            # -------------------------

            # --- Save 3D Gaussian assets (PLY) ---
            if output_gs and output_dir:
                try:
                    gs_path = os.path.join(output_dir, scene_name, f'sample_{actual_sample_idx:04d}', 'gaussians.ply')
                    gs_xyz = batch_recontrast_data.get('xyz_transformed', batch_recontrast_data['xyz'])[:1]
                    gs_rot = batch_recontrast_data.get('rot_maps_transformed', batch_recontrast_data['rot_maps'])[:1]
                    gs_sh = batch_recontrast_data.get('sh_maps_transformed', batch_recontrast_data['sh_maps'])[:1]
                    gs_scale = batch_recontrast_data['scale_maps'][:1]
                    gs_opacity = batch_recontrast_data['opacity_maps'][:1]
                    n_gs = save_gaussians_ply(gs_xyz, gs_rot, gs_scale, gs_opacity, gs_sh, gs_path)
                    print(f"GPU {gpu_id}: Saved {n_gs} Gaussians for sample {actual_sample_idx} -> gaussians.ply")
                except Exception as e:
                    print(f"GPU {gpu_id}: [WARN] Gaussian PLY output failed for sample {actual_sample_idx}: {e}")
            # -------------------------

            # Aggregate scene-level metrics (keep modes separate)
            scene_psnr_list.extend(recon_psnr + novel_psnr)
            scene_ssim_list.extend(recon_ssim + novel_ssim)
            scene_lpips_list.extend(recon_lpips + novel_lpips)
            

    # Calculate processing time
    scene_processing_time = time.time() - scene_start_time

    # Return scene results with separate reconstruction and novel view metrics
    lidar_metrics = {}
    if lidar_depth_l2_list:
        lidar_metrics = {
            'depth_l2': np.mean(lidar_depth_l2_list),
            'depth_l2_std': np.std(lidar_depth_l2_list),
            'depth_median_l2': np.mean(lidar_depth_median_l2_list),
            'depth_median_l2_std': np.std(lidar_depth_median_l2_list),
            'delta_1': np.mean(lidar_delta_1_list),
            'delta_2': np.mean(lidar_delta_2_list),
            'delta_3': np.mean(lidar_delta_3_list),
            'intensity_rmse': np.mean(lidar_intensity_rmse_list),
            'intensity_rmse_std': np.std(lidar_intensity_rmse_list),
            'ray_drop_acc': np.mean(lidar_ray_drop_acc_list),
            'ray_drop_acc_std': np.std(lidar_ray_drop_acc_list),
            'chamfer_distance': np.mean(lidar_chamfer_dist_list),
            'chamfer_distance_std': np.std(lidar_chamfer_dist_list),
        }
    return {
        'scene_idx': scene_batch.get('scene_idx', 0),
        'scene_name': scene_name,
        'scene_token': scene_token,
        'processed_samples': len(scene_psnr_list),
        'processing_time': scene_processing_time,
        'avg_sample_time': scene_processing_time / max(1, len(scene_psnr_list)),
        'gpu_id': gpu_id,
        'metrics': {
            'psnr': np.mean(scene_psnr_list) if scene_psnr_list else 0.0,
            'ssim': np.mean(scene_ssim_list) if scene_ssim_list else 0.0,
            'lpips': np.mean(scene_lpips_list) if scene_lpips_list else 0.0,
            'psnr_std': np.std(scene_psnr_list) if scene_psnr_list else 0.0,
            'ssim_std': np.std(scene_ssim_list) if scene_ssim_list else 0.0,
            'lpips_std': np.std(scene_lpips_list) if scene_lpips_list else 0.0,
            **lidar_metrics
        },
        'sample_metrics': {
            'psnr_list': scene_psnr_list,
            'ssim_list': scene_ssim_list,
            'lpips_list': scene_lpips_list
        },
        'recon_metrics': {
            'psnr_list': recon_psnr,
            'ssim_list': recon_ssim,
            'lpips_list': recon_lpips
        },
        'novel_metrics': {
            'psnr_list': novel_psnr,
            'ssim_list': novel_ssim,
            'lpips_list': novel_lpips
        }
    }


def save_lidar_cam_overlays(lidar_gt, lidar_out, batch_render_data, batch_splating_data,
                            scene_name, sample_idx, output_dir, num_cams=6, max_depth=80.0,
                            point_radius=1, alpha=0.8):
    """
    Project GT and Pred LiDAR points to each camera view and overlay on GT images.

    Args:
        lidar_gt: dict with 'raster_pts' [B,H,W,4], 'viewmat' [B,4,4], 'gt_depth' [B,H,W,1]
        lidar_out: dict with 'depth' [B,H,W,1] (predicted depth)
        batch_render_data: dict with ('e2c_extr', f, c) and ('K', f, c)
        batch_splating_data: dict with ('groudtruth', f, c) for GT camera images
        scene_name, sample_idx: for output path naming
        output_dir: root output directory
    """
    global_sample_idx = sample_idx
    lidar_cam_dir = os.path.join(output_dir, scene_name, f'sample_{global_sample_idx:04d}', 'lidar_cam')
    os.makedirs(lidar_cam_dir, exist_ok=True)

    H, W = lidar_gt['raster_pts'].shape[1:3]

    # Convert raster_pts (az, el, depth) → 3D points (x,y,z) in LiDAR frame
    # raster_pts[..., 0] = azimuth in degrees, [... ,1] = elevation in degrees
    raster_pts = lidar_gt['raster_pts']  # [B, H, W, 4]
    az_rad = torch.deg2rad(raster_pts[..., 0:1])   # [B, H, W, 1]
    el_rad = torch.deg2rad(raster_pts[..., 1:2])   # [B, H, W, 1]
    cos_el = torch.cos(el_rad)

    # GT depth [B, H, W, 1]
    gt_depth = lidar_gt['gt_depth']
    # Pred depth [B, H, W, 1]
    pred_depth = lidar_out['depth']

    # 3D points in LiDAR frame: [B, H, W, 3]
    gt_pts_3d = torch.cat([
        gt_depth * cos_el * torch.cos(az_rad),
        gt_depth * cos_el * torch.sin(az_rad),
        gt_depth * torch.sin(el_rad),
    ], dim=-1)

    pred_pts_3d = torch.cat([
        pred_depth * cos_el * torch.cos(az_rad),
        pred_depth * cos_el * torch.sin(az_rad),
        pred_depth * torch.sin(el_rad),
    ], dim=-1)

    # LiDAR → ego transform: inv(viewmat)
    # viewmat is ego_to_lidar, so lidar_to_ego = inv(viewmat)
    lidar_to_ego = torch.linalg.inv(lidar_gt['viewmat'])  # [B, 4, 4]

    # Helper to project points in LiDAR frame to a camera
    def _project_to_camera(pts_3d_lidar, e2c_extr, K, target_h, target_w):
        """pts_3d_lidar: [N, 3] in LiDAR frame → [M, 2] pixel coords and [M] depths"""
        device = pts_3d_lidar.device
        N = pts_3d_lidar.shape[0]

        # Force 2D [4, 4] matrices (handles [1,4,4], [4,4], or any shape with 16 elements)
        e2c = e2c_extr.reshape(4, 4)
        l2e = lidar_to_ego.reshape(4, 4)
        cam_T_lidar = torch.mm(e2c, l2e)  # [4, 4] lidar→camera transform

        # pts_h: [N, 4]
        ones = torch.ones(N, 1, device=device, dtype=pts_3d_lidar.dtype)
        pts_h = torch.cat([pts_3d_lidar, ones], dim=-1)

        # Transform: [N, 4] = [N, 4] @ [4, 4]
        pts_cam = torch.mm(pts_h, cam_T_lidar.T)  # [N, 4]
        pts_cam = pts_cam[:, :3]  # [N, 3]

        # Filter: in front of camera
        front = pts_cam[:, 2] > 0
        pts_front = pts_cam[front]

        if pts_front.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros(0)

        # Project with intrinsics (post-multiply to avoid .T on >2D)
        # uv_h = pts_front @ K.T  →  [M, 3] = [M, 3] @ [3, 3]
        uv_h = torch.mm(pts_front, K.T)  # [M, 3]
        uv_h[:, :2] /= uv_h[:, 2:3]
        u, v, z = uv_h[:, 0], uv_h[:, 1], uv_h[:, 2]

        in_bounds = (u >= 0) & (u < target_w) & (v >= 0) & (v < target_h)
        u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
        return torch.stack([u, v], dim=-1).cpu().numpy(), z.cpu().numpy()

    for cam_id in range(num_cams):
        frame_id = 0  # Use frame 0 (reconstruction frame)
        gt_key = ('groudtruth', frame_id, cam_id)
        e2c_key = ('e2c_extr', frame_id, cam_id)
        K_key = ('K', frame_id, cam_id)

        if gt_key not in batch_splating_data or e2c_key not in batch_render_data:
            continue

        # GT camera image [B, 3, H, W]
        gt_img_t = batch_splating_data[gt_key][0]  # [3, H, W]
        H_img, W_img = gt_img_t.shape[1], gt_img_t.shape[2]
        gt_np = gt_img_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # [H, W, 3]

        e2c_extr = batch_render_data[e2c_key][0:1]  # [1, 4, 4]
        K = batch_render_data[K_key][0, :3, :3]     # [3, 3]

        # Flatten Pred LiDAR points to [H*W, 3]
        pred_pts_flat = pred_pts_3d[0].reshape(-1, 3)

        # GT: use the raw sensor point cloud (full sweep, no range-image binning)
        # when available; fall back to the rasterized range-image points.
        if 'raw_points' in lidar_gt:
            gt_pts_valid = lidar_gt['raw_points'][0, :, :3]  # [N, 3] LiDAR frame
        else:
            gt_pts_flat = gt_pts_3d[0].reshape(-1, 3)
            gt_valid = lidar_gt['gt_depth'][0].reshape(-1) > 0
            gt_pts_valid = gt_pts_flat[gt_valid]

        pred_ray_drop = (torch.sigmoid(lidar_out['ray_drop_logits'][0]).reshape(-1) < 0.5)
        pred_valid = lidar_out['depth'][0].reshape(-1) > 0
        pred_valid = pred_valid

        pred_pts_valid = pred_pts_flat[pred_valid]

        # Project GT points
        gt_uv, gt_z = _project_to_camera(gt_pts_valid, e2c_extr, K, H_img, W_img)
        # Project Pred points
        pred_uv, pred_z = _project_to_camera(pred_pts_valid, e2c_extr, K, H_img, W_img)

        def _make_overlay(base_img, uv, z, path):
            """Overlay depth-colored points on base image and save."""
            overlay = base_img.copy()
            h, w = overlay.shape[:2]

            if len(uv) == 0:
                Image.fromarray((overlay * 255).astype(np.uint8)).save(path)
                return

            # Color by depth: blue(near) → green → red(far)
            depth_norm = np.clip(z / max_depth, 0, 1)
            colors = np.zeros((len(z), 3))
            colors[:, 0] = depth_norm
            colors[:, 1] = 1.0 - np.abs(depth_norm - 0.5) * 2
            colors[:, 2] = 1.0 - depth_norm
            colors = np.clip(colors, 0, 1)

            u_c = np.round(uv[:, 0]).astype(int)
            v_c = np.round(uv[:, 1]).astype(int)

            # Sort by depth descending (nearer on top)
            order = np.argsort(-z)
            u_c, v_c, colors = u_c[order], v_c[order], colors[order]

            for i in range(len(u_c)):
                u_pt, v_pt = u_c[i], v_c[i]
                col = colors[i]
                for dy in range(-point_radius, point_radius + 1):
                    for dx in range(-point_radius, point_radius + 1):
                        if dx*dx + dy*dy <= point_radius*point_radius:
                            ud, vd = u_pt + dx, v_pt + dy
                            if 0 <= ud < w and 0 <= vd < h:
                                overlay[vd, ud] = (1 - alpha) * overlay[vd, ud] + alpha * col

            Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(path)

        # Save GT overlay
        gt_out = os.path.join(lidar_cam_dir, f'cam_{cam_id}_gt_lidar_overlay.png')
        _make_overlay(gt_np, gt_uv, gt_z, gt_out)

        # Save Pred overlay (unfiltered)
        pred_out = os.path.join(lidar_cam_dir, f'cam_{cam_id}_pred_lidar_overlay.png')
        _make_overlay(gt_np, pred_uv, pred_z, pred_out)

        print(f"  GPU {0}: lidar_cam cam_{cam_id}: GT={len(gt_uv)}pts, "
              f"Pred={len(pred_uv)}pts")



def _apply_colormap(vals):
    """Map normalized values in [0,1] to a blue->green->red colormap.
    vals: [H, W] float -> RGB uint8 image [H, W, 3]
    """
    v = np.clip(np.asarray(vals, dtype=np.float32), 0, 1)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def render_bev_views(model, recontrast_data, device, scene_name, sample_idx, output_dir,
                     bev_x_range=50.0, bev_y_range=25.0, bev_res=0.2):
    """
    Render Bird's Eye View (BEV) images of the reconstructed 3DGS scene using a
    TRUE ORTHOGRAPHIC projection.

    A virtual camera looks straight down at the scene (parallel projection, no
    perspective foreshortening), so a tall object keeps the same footprint and
    proportions as its ground truth — no radial stretching, no center hole, and
    no moiré from a degenerate 90-degree perspective covariance (an affine
    projection maps a 3D Gaussian to an exact 2D Gaussian). The Gaussians are
    re-rendered from this camera to produce:
      - pred_RGB_bev.png    : color BEV image
      - pred_height_bev.png : height above ground, derived from rendered depth

    Implementation notes (orthographic = skip gsplat's perspective API):
      1. 3D covariances are built from (quat, scale) with `quat_scale_to_covar_preci`
         and transformed to the camera frame with `world_to_cam`.
      2. Projection is affine:  u = fx * x_cam + cx,  v = fy * y_cam + cy,
         with fx = W / (2*bev_y_range) px/m, fy = H / (2*bev_x_range) px/m, and
         2D covariance  covars2d = J * covars_cam * J^T,  J = diag(fx, fy).
         (For perspective the same would require dividing by z, which blows up
         when looking straight down.)
      3. The remaining pipeline (conics / radius / isect_tiles /
         isect_offset_encode / rasterize_to_pixels) is identical to what the
         high-level `rasterization()` uses internally, so rendering is exactly
         as faithful as the normal perspective path.

    Args:
        bev_x_range: longitudinal range in meters around the ego (+/- forward/back)
        bev_y_range: lateral range in meters around the ego (+/- left/right)
        bev_res: BEV pixel resolution (m/pixel)

    Returns the list of saved file paths.
    """
    # Image size: columns span the lateral (y) range, rows span the longitudinal (x)
    # range, both at `bev_res` m/pixel (consistent with save_lidar_bev)
    bev_w = int(2 * bev_y_range / bev_res)
    bev_h = int(2 * bev_x_range / bev_res)
    bev_dir = os.path.join(output_dir, scene_name, f'sample_{sample_idx:04d}', 'bev')
    os.makedirs(bev_dir, exist_ok=True)

    # Use the unified ego-frame Gaussians (same as render_splating_imgs uses)
    xyz = recontrast_data.get('xyz_transformed', recontrast_data['xyz'])[:1].squeeze(0)      # [N, 3]
    rot = recontrast_data.get('rot_maps_transformed', recontrast_data['rot_maps'])[:1].squeeze(0)  # [N, 4]
    sh = recontrast_data.get('sh_maps_transformed', recontrast_data['sh_maps'])[:1].squeeze(0)      # [N, K, 3]
    scale = recontrast_data['scale_maps'][:1].squeeze(0)                                        # [N, 3]
    opacity = recontrast_data['opacity_maps'][:1].squeeze(0).squeeze(-1)                        # [N]
    N = xyz.shape[0]

    # --- BEV virtual camera (ego -> bev-cam extrinsic) ---
    # Camera axes in ego frame:
    #   x_cam = -y_ego (image right = ego -y), y_cam = -x_ego (image up = ego +x),
    #   z_cam = -z_ego (looking down). det(R) = +1 (right-handed, no mirroring).
    # `cam_depth_ref` only sets the offset of the camera-space depth
    # (depth = cam_depth_ref - z_ego); orthographic projection is invariant to
    # it, so any value above the tallest object works.
    cam_depth_ref = 100.0
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[0, 0], viewmat[0, 1] = 0.0, -1.0
    viewmat[1, 0], viewmat[1, 1] = -1.0, 0.0
    viewmat[2, 2] = -1.0
    viewmat[2, 3] = cam_depth_ref  # t = -R @ cam_pos = (0, 0, cam_depth_ref)
    viewmat = viewmat.unsqueeze(0)  # [1, 4, 4]

    # --- Orthographic scale (pixels per meter) ---
    fx = bev_w / (2.0 * bev_y_range)  # px/m along x_cam (-y_ego)
    fy = bev_h / (2.0 * bev_x_range)  # px/m along y_cam (-x_ego)
    cx = bev_w / 2.0
    cy = bev_h / 2.0

    # 1. 3D covariances in world (ego) frame, then to camera frame
    covars, _ = quat_scale_to_covar_preci(rot, scale, compute_preci=False)  # [N, 3, 3]
    means_c, covars_c = world_to_cam(xyz, covars, viewmat)  # [1, N, 3], [1, N, 3, 3]

    # 2. Orthographic projection (affine => exact 2D Gaussian, no depth division)
    means2d = torch.stack(
        [means_c[..., 0] * fx + cx, means_c[..., 1] * fy + cy], dim=-1
    )  # [1, N, 2]
    # J = [[fx, 0, 0], [0, fy, 0]];  covars2d = J * covars_c * J^T is just a rescale
    covars2d = torch.stack(
        [
            fx * fx * covars_c[..., 0, 0],
            fx * fy * covars_c[..., 0, 1],
            fx * fy * covars_c[..., 1, 0],
            fy * fy * covars_c[..., 1, 1],
        ],
        dim=-1,
    ).reshape(1, N, 2, 2)  # [1, N, 2, 2]

    # 3. Conics / radius / depths / validity (same convention as fully_fused_projection)
    eps2d = 0.3
    covars2d = covars2d + torch.eye(2, device=device, dtype=torch.float32) * eps2d
    det = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    det = det.clamp(min=1e-10)
    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [1, N, 3]

    depths = means_c[..., 2]  # [1, N], camera-space z = cam_depth_ref - z_ego
    radius = 3.0 * torch.sqrt(
        torch.stack([covars2d[..., 0, 0], covars2d[..., 1, 1]], dim=-1)
    )  # [1, N, 2]
    valid = (det > 0) & (depths > 0.01) & (depths < 1e10)
    radius[~valid] = 0.0
    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < bev_w)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < bev_h)
    )
    radius[~inside] = 0.0
    radii = radius.int()  # [1, N, 2]

    # 4. Identify intersecting tiles and rasterize (same kernels as rasterization())
    tile_size = 16
    tile_width = (bev_w + tile_size - 1) // tile_size
    tile_height = (bev_h + tile_size - 1) // tile_size
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, packed=False,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    # 5. Colors: activate SH for the BEV view (same as rendering.py non-packed branch)
    sh_degree = getattr(model, 'sh_degree', 3)
    camtoworlds = torch.inverse(viewmat)  # [1, 4, 4]
    dirs = xyz[None, :, :] - camtoworlds[:, None, :3, 3]  # [1, N, 3]
    masks = radii[..., 0] > 0  # [1, N]
    shs = sh.unsqueeze(0).expand(1, -1, -1, -1)  # [1, N, K, 3]
    colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [1, N, 3]
    colors = torch.clamp_min(colors + 0.5, 0.0)
    colors = torch.cat([colors, depths[..., None]], dim=-1)  # [1, N, 4] (RGB+D)

    opacities = opacity[None, :].repeat(1, 1)  # [1, N]
    pix_vels = torch.zeros(1, N, 2, device=device, dtype=torch.float32)

    render_colors, render_alphas, _ = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        pix_vels,
        bev_w,
        bev_h,
        tile_size,
        isect_offsets,
        flatten_ids,
        rolling_shutter_time=None,
        backgrounds=None,
        packed=False,
    )  # render_colors [1, bev_h, bev_w, 4]

    saved_paths = []
    mask = (render_alphas[0].squeeze(-1) > 0.01).cpu().numpy()  # [H, W], pixels with rendered content

    # Color BEV
    rgb_img = render_colors[..., :3].permute(0, 3, 1, 2)[0]  # [3, H, W]
    rgb_img = rgb_img.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    rgb_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
    rgb_img[~mask] = 255  # white background where nothing was rendered
    rgb_path = os.path.join(bev_dir, 'pred_RGB_bev.png')
    Image.fromarray(rgb_img).save(rgb_path)
    saved_paths.append(rgb_path)

    # Height BEV: depth is camera-space z = cam_depth_ref - z_ego,
    # so height above ground = cam_depth_ref - depth = z_ego
    depth = render_colors[..., 3]  # [1, H, W]
    height = (cam_depth_ref - depth).clamp(min=0.0)
    max_height = 10.0  # colormap scale for the height map
    height_norm = (height[0] / max_height).clamp(0, 1).cpu().numpy()  # [H, W]
    height_img = _apply_colormap(height_norm)
    height_img[~mask] = 255
    height_path = os.path.join(bev_dir, 'pred_height_bev.png')
    Image.fromarray(height_img).save(height_path)
    saved_paths.append(height_path)

    return saved_paths


def save_gaussians_ply(xyz, rot, scale, opacity, sh, path):
    """Save the 3D Gaussian assets of a scene to a standard 3DGS binary PLY file.

    Args:
        xyz: [B, N, 3] Gaussian centers (world/ego frame).
        rot: [B, N, 4] quaternions (w, x, y, z).
        scale: [B, N, 3] Gaussian scales.
        opacity: [B, N, 1] opacities (0-1).
        sh: [B, N, K, 3] spherical harmonic coefficients.
        path: Output .ply path.

    The PLY follows the standard 3DGS layout (binary little-endian float32):
    x y z nx ny nz f_dc_0..2 f_rest_0..(K*3-4) opacity scale_0..2 rot_0..3.

    Returns the number of Gaussians written.
    """
    xyz_np = xyz[0].reshape(-1, 3).detach().cpu().numpy().astype(np.float32)
    rot_np = rot[0].reshape(-1, 4).detach().cpu().numpy().astype(np.float32)
    scale_np = scale[0].reshape(-1, 3).detach().cpu().numpy().astype(np.float32)
    opacity_np = opacity[0].reshape(-1, 1).detach().cpu().numpy().astype(np.float32)
    K = sh[0].shape[-2]
    sh_np = sh[0].reshape(-1, K, 3).detach().cpu().numpy().astype(np.float32)
    N = xyz_np.shape[0]
    f_dc = sh_np[:, 0, :]  # [N, 3]
    f_rest = sh_np[:, 1:, :].reshape(N, -1)  # [N, (K-1)*3]
    rest_num = f_rest.shape[1]
    normals = np.zeros_like(xyz_np)
    data = np.concatenate([xyz_np, normals, f_dc, f_rest, opacity_np, scale_np, rot_np], axis=1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
        for i in range(rest_num):
            f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write(b"end_header\n")
        f.write(data.tobytes())
    return N


def save_lidar_ply(lidar_gt, lidar_out, scene_name, sample_idx, output_dir):
    """Save GT and predicted LiDAR point clouds as PLY files in the lidar folder.

    When the raw sensor point cloud ('raw_points') is available, the GT PLY is
    written from the full raw sweep (no range-image binning); otherwise it falls
    back to the points recovered from the range image. Points are transformed
    into the ego frame (same convention as save_lidar_bev) and written as ASCII
    PLY with x/y/z/intensity/depth properties to <sample>/lidar/gt_points.ply
    and <sample>/lidar/pred_points.ply. A third file
    <sample>/lidar/pred_points_mask.ply holds the predicted points additionally
    masked by the GT depth>0 validity mask (for fair comparison on the same rays).

    Returns the list of saved file paths.
    """
    lidar_dir = os.path.join(output_dir, scene_name, f'sample_{sample_idx:04d}', 'lidar')
    os.makedirs(lidar_dir, exist_ok=True)

    raster_pts = lidar_gt['raster_pts']  # [B, H, W, 4]: [az(deg), el(deg), _, _]
    az_rad = torch.deg2rad(raster_pts[..., 0:1])  # [B, H, W, 1]
    el_rad = torch.deg2rad(raster_pts[..., 1:2])  # [B, H, W, 1]
    cos_el = torch.cos(el_rad)
    lidar_to_ego = torch.linalg.inv(lidar_gt['viewmat']).reshape(1, 4, 4)

    def _ego_points(depth):
        pts_lidar = torch.cat([
            depth * cos_el * torch.cos(az_rad),
            depth * cos_el * torch.sin(az_rad),
            depth * torch.sin(el_rad),
        ], dim=-1)  # [B, H, W, 3]
        ones = torch.ones_like(pts_lidar[..., :1])
        pts_h = torch.cat([pts_lidar, ones], dim=-1)  # [B, H, W, 4]
        return torch.matmul(pts_h, lidar_to_ego.transpose(1, 2))[..., :3]  # [B, H, W, 3]

    def _write_ply(path, pts, depth, intensity, valid_extra=None):
        pts_np = pts[0].reshape(-1, 3).detach().cpu().numpy()
        d_np = depth[0].reshape(-1).detach().cpu().numpy()
        i_np = intensity[0].reshape(-1).detach().cpu().numpy()
        valid = d_np > 0
        if valid_extra is not None:
            valid = valid & valid_extra
        pts_np, d_np, i_np = pts_np[valid], d_np[valid], i_np[valid]
        n = pts_np.shape[0]
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("property float depth\n")
            f.write("end_header\n")
            for i in range(n):
                f.write(
                    f"{pts_np[i, 0]:.4f} {pts_np[i, 1]:.4f} {pts_np[i, 2]:.4f} "
                    f"{i_np[i]:.4f} {d_np[i]:.4f}\n"
                )
        return n

    saved_paths = []
    if 'raw_points' in lidar_gt:
        # GT from the raw sensor point cloud (full sweep, no range-image binning)
        pts_lidar = lidar_gt['raw_points'][0, :, :3]  # [N, 3] LiDAR frame
        depth_lidar = torch.linalg.norm(pts_lidar, dim=-1, keepdim=True)  # [N, 1]
        int_lidar = (lidar_gt['raw_points'][0, :, 3:4] / 255.0)  # [N, 1]
        ones = torch.ones_like(pts_lidar[..., :1])
        pts_h = torch.cat([pts_lidar, ones], dim=-1)  # [N, 4]
        gt_pts = torch.matmul(pts_h, lidar_to_ego.transpose(1, 2))[..., :3][None]  # [1, N, 3] ego frame
        gt_depth_v = depth_lidar[None]  # [1, N, 1]
        gt_int_v = int_lidar[None]  # [1, N, 1]
    else:
        gt_pts = _ego_points(lidar_gt['gt_depth'])
        gt_depth_v = lidar_gt['gt_depth']
        gt_int_v = lidar_gt['gt_intensity']
    gt_path = os.path.join(lidar_dir, 'gt_points.ply')
    _write_ply(gt_path, gt_pts, gt_depth_v, gt_int_v)
    saved_paths.append(gt_path)

    pred_pts = _ego_points(lidar_out['depth'])
    pred_path = os.path.join(lidar_dir, 'pred_points.ply')
    _write_ply(pred_path, pred_pts, lidar_out['depth'], lidar_out['intensity'])
    saved_paths.append(pred_path)

    # Pred points additionally masked by GT depth>0 (same rays as GT)
    gt_valid = (lidar_gt['gt_depth'][0].reshape(-1).detach().cpu().numpy() > 0)
    pred_mask_path = os.path.join(lidar_dir, 'pred_points_mask.ply')
    _write_ply(pred_mask_path, pred_pts, lidar_out['depth'], lidar_out['intensity'], valid_extra=gt_valid)
    saved_paths.append(pred_mask_path)

    return saved_paths


def save_lidar_bev(lidar_gt, lidar_out, scene_name, sample_idx, output_dir,
                   bev_x_range=50.0, bev_y_range=25.0, bev_res=0.2, max_depth=80.0):
    """
    Project GT and predicted LiDAR points onto a Bird's Eye View (XY) grid and save images.

    Points (azimuth/elevation/depth) in the LiDAR frame are converted to the ego
    frame and binned into a 2D grid covering
    [-bev_x_range, +bev_x_range] x [-bev_y_range, +bev_y_range]. For each bin the
    closest point (smallest range) is kept. Points are drawn on a pure white
    background and colored with a blue->green->red value colormap (depth maps
    colored by range in [0, max_depth], intensity maps by intensity in [0, 1]):
      - lidar_bev_gt_depth.png / lidar_bev_pred_depth.png
      - lidar_bev_gt_intensity.png / lidar_bev_pred_intensity.png

    Returns the list of saved file paths.
    """
    bev_h = int(2 * bev_x_range / bev_res)
    bev_w = int(2 * bev_y_range / bev_res)
    bev_dir = os.path.join(output_dir, scene_name, f'sample_{sample_idx:04d}', 'bev')
    os.makedirs(bev_dir, exist_ok=True)

    raster_pts = lidar_gt['raster_pts']  # [B, H, W, 4]
    az_rad = torch.deg2rad(raster_pts[..., 0:1])  # [B, H, W, 1]
    el_rad = torch.deg2rad(raster_pts[..., 1:2])
    cos_el = torch.cos(el_rad)

    # viewmat is ego->lidar, so lidar_to_ego = inv(viewmat)
    lidar_to_ego = torch.linalg.inv(lidar_gt['viewmat']).reshape(1, 4, 4)

    def _ego_points(depth):
        """depth: [B, H, W, 1] -> 3D points in ego frame [B, H, W, 3]"""
        pts_lidar = torch.cat([
            depth * cos_el * torch.cos(az_rad),
            depth * cos_el * torch.sin(az_rad),
            depth * torch.sin(el_rad),
        ], dim=-1)
        ones = torch.ones_like(pts_lidar[..., :1])
        pts_h = torch.cat([pts_lidar, ones], dim=-1)  # [B, H, W, 4]
        return torch.matmul(pts_h, lidar_to_ego.transpose(1, 2))[..., :3]

    def _scatter_bev(pts_ego, depth, values, path, vmin, vmax, stretch_sqrt=False):
        """Bin points into the BEV grid, keep closest per bin.

        Pure white background; points are colored with the blue->green->red
        value colormap (depth maps: colored by range in [0, max_depth];
        intensity maps: colored by intensity in [0, 1]). When `stretch_sqrt`
        is True the normalized value is sqrt-stretched, which spreads the
        usually-crowded near-range depths across more colors.
        """
        pts = pts_ego[0].reshape(-1, 3).detach().cpu().numpy()
        d = depth[0].reshape(-1).detach().cpu().numpy()
        v = values[0].reshape(-1).detach().cpu().numpy()
        valid = d > 0
        pts, d, v = pts[valid], d[valid], v[valid]
        img = np.zeros((bev_h, bev_w), dtype=np.float32)
        if len(pts) > 0:
            x, y = pts[:, 0], pts[:, 1]
            in_range = (np.abs(x) <= bev_x_range) & (np.abs(y) <= bev_y_range)
            x, y, d, v = x[in_range], y[in_range], d[in_range], v[in_range]
            # row 0 = far front (+x), col 0 = -y side
            rows = np.floor((bev_x_range - x) / bev_res).astype(np.int64)
            cols = np.floor((y + bev_y_range) / bev_res).astype(np.int64)
            rows = np.clip(rows, 0, bev_h - 1)
            cols = np.clip(cols, 0, bev_w - 1)
            flat = rows * bev_w + cols
            # Keep the closest (smallest range) point per bin
            order = np.argsort(d, kind='stable')
            s_flat, s_v = flat[order], v[order]
            _, first = np.unique(s_flat, return_index=True)
            img.reshape(-1)[s_flat[first]] = s_v[first]
        colored = np.full((bev_h, bev_w, 3), 255, dtype=np.uint8)  # pure white background
        norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        if stretch_sqrt:
            norm = np.sqrt(norm)
        cm = _apply_colormap(norm)
        pts_mask = img > 0
        colored[pts_mask] = cm[pts_mask]
        Image.fromarray(colored).save(path)

    gt_pts = _ego_points(lidar_gt['gt_depth'])
    pred_pts = _ego_points(lidar_out['depth'])

    saved_paths = []
    tasks = [
        ('lidar_bev_gt_depth.png', gt_pts, lidar_gt['gt_depth'], lidar_gt['gt_depth'], 0.0, max_depth, True),
        ('lidar_bev_pred_depth.png', pred_pts, lidar_out['depth'], lidar_out['depth'], 0.0, max_depth, True),
        ('lidar_bev_gt_intensity.png', gt_pts, lidar_gt['gt_depth'], lidar_gt['gt_intensity'], 0.0, 1.0, False),
        ('lidar_bev_pred_intensity.png', pred_pts, lidar_out['depth'], lidar_out['intensity'], 0.0, 1.0, False),
    ]
    for fname, pts, depth, values, vmin, vmax, stretch in tasks:
        path = os.path.join(bev_dir, fname)
        _scatter_bev(pts, depth, values, path, vmin, vmax, stretch_sqrt=stretch)
        saved_paths.append(path)
    return saved_paths


def _run_single_gpu_inference(model, scene_dataloader, device, save_results=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518',
                              bev_x_range=50.0, bev_y_range=25.0, bev_resolution=0.2, output_gs=False):
    """Run inference on all scenes - simplified using unified scene processing"""
    print(f"\nStarting scene-based inference on device: {device}")
    print(f"Number of scenes: {len(scene_dataloader)}")
    all_scene_results = []
    overall_psnr, overall_ssim, overall_lpips = [], [], []
    recon_psnr, recon_ssim, recon_lpips = [], [], []
    novel_psnr, novel_ssim, novel_lpips = [], [], []

    with torch.no_grad():
        for scene_idx, scene_batch in enumerate(scene_dataloader):

            # Keep the scene's own scene_idx from the data module (real index into
            # the dataset's scene list). Only fall back to the enumeration index
            # when the batch does not carry one (e.g. pre-loaded samples).
            scene_batch.setdefault('scene_idx', scene_idx)

            result = _process_scene_batch(model, scene_batch, device, gpu_id=0,
                                        save_renders=save_results, output_dir=output_dir,
                                        novel_distances=novel_distances, eval_resolution=eval_resolution,
                                        batch_idx=scene_idx, bev_x_range=bev_x_range,
                                        bev_y_range=bev_y_range, bev_resolution=bev_resolution,
                                        output_gs=output_gs)

            all_scene_results.append(result)
            overall_psnr.extend(result['sample_metrics']['psnr_list'])
            overall_ssim.extend(result['sample_metrics']['ssim_list'])
            overall_lpips.extend(result['sample_metrics']['lpips_list'])

            # Collect separate reconstruction and novel view metrics
            recon_psnr.extend(result['recon_metrics']['psnr_list'])
            recon_ssim.extend(result['recon_metrics']['ssim_list'])
            recon_lpips.extend(result['recon_metrics']['lpips_list'])
            novel_psnr.extend(result['novel_metrics']['psnr_list'])
            novel_ssim.extend(result['novel_metrics']['ssim_list'])
            novel_lpips.extend(result['novel_metrics']['lpips_list'])

    # Aggregate LiDAR metrics across scenes
    lidar_depth_l2_all, lidar_depth_median_l2_all = [], []
    lidar_delta_1_all, lidar_delta_2_all, lidar_delta_3_all = [], [], []
    lidar_intensity_rmse_all, lidar_ray_drop_acc_all, lidar_chamfer_dist_all = [], [], []
    for r in all_scene_results:
        m = r.get('metrics', {})
        if 'depth_l2' in m:
            lidar_depth_l2_all.append(m['depth_l2'])
            lidar_depth_median_l2_all.append(m['depth_median_l2'])
            lidar_delta_1_all.append(m['delta_1'])
            lidar_delta_2_all.append(m['delta_2'])
            lidar_delta_3_all.append(m['delta_3'])
            lidar_intensity_rmse_all.append(m['intensity_rmse'])
            lidar_ray_drop_acc_all.append(m['ray_drop_acc'])
            lidar_chamfer_dist_all.append(m['chamfer_distance'])

    # Print final results - separate for reconstruction and novel view
    final_psnr = np.mean(overall_psnr) if overall_psnr else 0.0
    final_ssim = np.mean(overall_ssim) if overall_ssim else 0.0
    final_lpips = np.mean(overall_lpips) if overall_lpips else 0.0

    final_recon_psnr = np.mean(recon_psnr) if recon_psnr else 0.0
    final_recon_ssim = np.mean(recon_ssim) if recon_ssim else 0.0
    final_recon_lpips = np.mean(recon_lpips) if recon_lpips else 0.0

    final_novel_psnr = np.mean(novel_psnr) if novel_psnr else 0.0
    final_novel_ssim = np.mean(novel_ssim) if novel_ssim else 0.0
    final_novel_lpips = np.mean(novel_lpips) if novel_lpips else 0.0

    print(f"\n{'='*60}")
    print(f"SINGLE-GPU INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"Scenes: {len(all_scene_results)}, Samples: {sum(r['processed_samples'] for r in all_scene_results)}")
    print(f"Overall PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    print(f"\nScene Reconstruction (Frame 0):")
    print(f"  PSNR: {final_recon_psnr:.4f}, SSIM: {final_recon_ssim:.4f}, LPIPS: {final_recon_lpips:.4f}")
    print(f"\nNovel View Synthesis (Middle Frames):")
    print(f"  PSNR: {final_novel_psnr:.4f}, SSIM: {final_novel_ssim:.4f}, LPIPS: {final_novel_lpips:.4f}")
    if lidar_depth_l2_all:
        print(f"\nLiDAR Metrics (scenes: {len(lidar_depth_l2_all)}):")
        print(f"  Depth L2: {np.mean(lidar_depth_l2_all):.4f} ± {np.std(lidar_depth_l2_all):.4f}")
        print(f"  Median Depth L2: {np.mean(lidar_depth_median_l2_all):.4f} ± {np.std(lidar_depth_median_l2_all):.4f}")
        print(f"  δ1: {np.mean(lidar_delta_1_all):.4f}, δ2: {np.mean(lidar_delta_2_all):.4f}, δ3: {np.mean(lidar_delta_3_all):.4f}")
        print(f"  Intensity RMSE: {np.mean(lidar_intensity_rmse_all):.4f} ± {np.std(lidar_intensity_rmse_all):.4f}")
        print(f"  Ray Drop Acc: {np.mean(lidar_ray_drop_acc_all):.4f} ± {np.std(lidar_ray_drop_acc_all):.4f}")
        print(f"  Chamfer Distance: {np.mean(lidar_chamfer_dist_all):.4f} ± {np.std(lidar_chamfer_dist_all):.4f}")
    
    # Save results
    # if save_results and output_dir:
    if output_dir:
        lidar_overall = {}
        if lidar_depth_l2_all:
            lidar_overall = {
                'depth_l2': np.mean(lidar_depth_l2_all),
                'depth_l2_std': np.std(lidar_depth_l2_all),
                'depth_median_l2': np.mean(lidar_depth_median_l2_all),
                'depth_median_l2_std': np.std(lidar_depth_median_l2_all),
                'delta_1': np.mean(lidar_delta_1_all),
                'delta_2': np.mean(lidar_delta_2_all),
                'delta_3': np.mean(lidar_delta_3_all),
                'intensity_rmse': np.mean(lidar_intensity_rmse_all),
                'intensity_rmse_std': np.std(lidar_intensity_rmse_all),
                'ray_drop_acc': np.mean(lidar_ray_drop_acc_all),
                'ray_drop_acc_std': np.std(lidar_ray_drop_acc_all),
                'chamfer_distance': np.mean(lidar_chamfer_dist_all),
                'chamfer_distance_std': np.std(lidar_chamfer_dist_all),
            }
        final_results = {
            'overall_metrics': {
                'psnr': final_psnr, 'ssim': final_ssim, 'lpips': final_lpips,
                'psnr_std': np.std(overall_psnr), 'ssim_std': np.std(overall_ssim), 'lpips_std': np.std(overall_lpips),
                **lidar_overall
            },
            'scene_results': all_scene_results
        }
        save_inference_results(final_results, output_dir)

    return all_scene_results


def save_inference_results(results, output_dir):
    """Save inference results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'overall_metrics' in results:
        # New format with overall metrics
        scene_results = results['scene_results']
        overall_metrics = results['overall_metrics']
        
        # Create summary
        summary = {
            'overall_metrics': overall_metrics,
            'total_scenes': len(scene_results),
            'total_samples': sum(r['processed_samples'] for r in scene_results),
            'total_time': sum(r['processing_time'] for r in scene_results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'metrics': r['metrics']
                }
                for r in scene_results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save per-scene evaluation results
        for scene_result in scene_results:
            scene_name = scene_result['scene_name']
            scene_eval_file = os.path.join(output_dir, f'scene_{scene_name}_evaluation.json')
            scene_eval_data = {
                'scene_name': scene_name,
                'scene_token': scene_result['scene_token'],
                'metrics': scene_result['metrics'],
                'sample_metrics': scene_result['sample_metrics'],
                'processing_info': {
                    'processed_samples': scene_result['processed_samples'],
                    'processing_time': scene_result['processing_time'],
                    'avg_sample_time': scene_result['avg_sample_time']
                }
            }
            
            with open(scene_eval_file, 'w') as f:
                json.dump(scene_eval_data, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")
        print(f"  Per-scene evaluations: {output_dir}/scene_*_evaluation.json")
        
    else:
        # Legacy format
        summary = {
            'total_scenes': len(results),
            'total_samples': sum(r['processed_samples'] for r in results),
            'total_time': sum(r['processing_time'] for r in results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'scene_stats': r.get('scene_stats', {})
                }
                for r in results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")


def main():
    parser = argparse.ArgumentParser(description='Scene-based inference for VGGT3DGS')
    parser.add_argument('--cfg_path', type=str, required=True, help='Configuration file path')
    parser.add_argument('--restore_ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum number of scenes to process (default: all scenes)')
    parser.add_argument('--scene', type=str, default=None, help='Specific scene name to process (e.g., scene-0061). If set, only this scene is processed.')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cuda:0)')

    parser.add_argument('--no_renders', action='store_true', help='Disable saving rendered images and novel views')
    parser.add_argument('--novel_distances', type=str, default='0.5,1.0,2.0,3.0', 
                       help='Novel view translation distances in meters (comma-separated, e.g., "0.5,1.0,2.0,3.0")')
    parser.add_argument('--eval_resolution', type=str, default='original',# choices=['original', 'upsampled'],
                       help='Evaluation resolution mode: "original" for 280x518, "upsampled" for 900x1600')
    parser.add_argument('--bev_x_range', type=float, default=50.0,
                       help='BEV longitudinal range (+/- meters around ego, forward/back)')
    parser.add_argument('--bev_y_range', type=float, default=25.0,
                       help='BEV lateral range (+/- meters around ego, left/right)')
    parser.add_argument('--bev_resolution', type=float, default=0.2,
                       help='BEV pixel resolution (meters per pixel)')
    parser.add_argument('--gs', action='store_true',
                       help='Save the scene 3D Gaussian assets as a PLY file '
                            '(per sample: <sample>/gaussians.ply, ~600MB each)')

    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.cfg_path}")
 
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set batch_size to 1 for inference (CRITICAL: must be 1 for proper scene processing)
    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size'] = 1
    
    # Pass temporal config from data_cfg to model_cfg for time_delta calculation
    if 'context_span' in config['data_cfg']:
        config['model_cfg']['context_span'] = config['data_cfg']['context_span']
    if 'nuscenes_version' in config['data_cfg']:
        config['model_cfg']['nuscenes_version'] = config['data_cfg']['nuscenes_version']


    # Parse device
    if args.device:
        # Single GPU specified: "cuda:0" or "0"
        if args.device.startswith('cuda:'):
            device = args.device
        else:
            device = f"cuda:{args.device}"
    elif config.get('devices'):
        device = f"cuda:{config['devices'][0]}"
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Batch size: {config['data_cfg']['batch_size']}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config['save_dir'], 'scene_inference_results')
    
    print(f"Output directory: {args.output_dir}")
    
    # Parse save renders flag
    save_renders = not args.no_renders
    
    # Parse novel view distances
    try:
        novel_distances = [float(d.strip()) for d in args.novel_distances.split(',')]
    except ValueError:
        raise ValueError(f"Invalid novel_distances format: {args.novel_distances}. Use comma-separated floats like '0.5,1.0,2.0,3.0'")
    
    print(f"Save renders: {save_renders}")
    print(f"Novel view distances: {novel_distances}")
    print(f"Evaluation resolution: {args.eval_resolution}")
    print(f"BEV range: x +/-{args.bev_x_range}m, y +/-{args.bev_y_range}m (orthographic), res {args.bev_resolution}m/px")
    print(f"Save Gaussian assets (PLY): {args.gs}")

    # CRITICAL: Ensure batch_size is 1 before creating data module
    print(f"Original batch_size in config: {config['data_cfg'].get('batch_size', 'not set')}")
    config['data_cfg']['batch_size'] = 1  # Must be 1 for proper scene processing
    print(f"Override batch_size to: {config['data_cfg']['batch_size']}")

    # Initialize scene-based data module
    print("Initializing scene-based data module...")
    data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    
    # Get scene dataloader
    scene_dataloader = data_module.test_scene_dataloader()
    total_scenes = len(scene_dataloader)
    
    if args.scene:
        print(f"Filtering to scene: {args.scene}")
        scene_list = []
        for scene_batch in scene_dataloader:
            if scene_batch['scene_name'] == args.scene:
                scene_list.append(scene_batch)
                break
        if not scene_list:
            print(f"[ERROR] Scene '{args.scene}' not found in dataset!")
            sys.exit(1)
        scene_dataloader = scene_list
    elif args.max_scenes:
        print(f"Limiting to {args.max_scenes} scenes (out of {total_scenes})")
        scene_list = []
        for i, scene_batch in enumerate(scene_dataloader):
            if i >= args.max_scenes:
                break
            scene_list.append(scene_batch)
        scene_dataloader = scene_list
    else:
        print(f"Processing all {total_scenes} scenes")


    # Run single-GPU inference
    results = run_inference(
        model_cfg=config['model_cfg'],
        checkpoint_path=args.restore_ckpt,
        scene_dataloader=scene_dataloader,
        device=device,
        save_results=save_renders,
        output_dir=args.output_dir,
        novel_distances=novel_distances,
        eval_resolution=args.eval_resolution,
        bev_x_range=args.bev_x_range,
        bev_y_range=args.bev_y_range,
        bev_resolution=args.bev_resolution,
        output_gs=args.gs,
    )
    
    print(f"\nScene-based inference completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()