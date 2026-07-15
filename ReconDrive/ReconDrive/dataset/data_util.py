#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import os

import numpy as np
import PIL.Image as pil
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from dataset.types import *
from dataset.augmentations import random_crop_borders, crop_sample, resize_sample, duplicate_sample,  colorjitter_sample, to_tensor_sample


_DEL_KEYS= ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'intrinsics', 'extrinsics', 'contexts', 'splitname'] 

_NP_KEYS = ['c2e_extr','K','cam_T_cam','ego_T_ego','ego_pose','timestamp']
def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    image_shape = data_transform.keywords['image_shape']
    resize_transform = transforms.Resize(image_shape, interpolation=pil.LANCZOS)
    sample['mask'] = resize_transform(sample['mask'])
    tensor_transform = transforms.ToTensor()    
    sample['mask'] = tensor_transform(sample['mask'])
    return sample

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    if len(sample) == 1:
        single_sample = sample[0]
        if 'timestamp' in single_sample and not isinstance(single_sample['timestamp'], np.ndarray):
            single_sample['timestamp'] = np.array([single_sample['timestamp']], dtype=np.int64)
        return single_sample

    stacked_sample = {}
    for key in sample[0]:
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token', 'scene_token', 'scene_name', 'scene_idx']:
            stacked_sample[key] = sample[0][key]
        elif key == 'vehicle_annotations':
            if all(key in s for s in sample):
                stacked_sample[key] = [s[key] for s in sample]
        elif key == 'timestamp':
            stacked_sample[key] = np.array([s[key] for s in sample], dtype=np.int64)
        else:
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            elif is_pilimg(sample[0][key]):
                tensor_transform = transforms.ToTensor()   
                stacked_sample[key] = torch.stack([tensor_transform(s[key]) for s in sample],0)
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                if len(sample[0][key]) > 0:
                    if is_tensor(sample[0][key][0]):
                        for i in range(len(sample[0][key])):
                            stacked_sample[key].append(
                                torch.stack([s[key][i] for s in sample], 0))
                    elif is_numpy(sample[0][key][0]):
                        for i in range(len(sample[0][key])):
                            stacked_sample[key].append(
                                np.stack([s[key][i] for s in sample], 0))
                else:
                    stacked_sample[key] = []

    return stacked_sample


def img_loader(path):
    """
    This function loads rgb image.
    """
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))    
    with open(fname, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('L')


def inference_transforms(sample, image_shape):
    """
    Minimal transformations for inference - only essential operations
    """
    sample = duplicate_sample(sample)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = to_tensor_sample(sample)
    return sample

def train_transforms(sample, image_shape, crop_scale=[],crop_ratio=[],crop_prob=1.0,jittering=[],jittering_prob=1.0):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    crop_train_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_scale) > 0:
        borders = random_crop_borders(sample['rgb'].size[::-1], crop_scale,crop_ratio)
        sample = crop_sample(sample, borders,prob=crop_prob)
    sample = duplicate_sample(sample)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)

    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering, prob=jittering_prob)
    sample = to_tensor_sample(sample)
    return sample

def inference_transforms(sample, image_shape):
    """
    Minimal transformations for inference - only essential operations
    """
    sample = duplicate_sample(sample)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = to_tensor_sample(sample)
    return sample

def align_dataset(sample, contexts=None):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    
    aug_contexts = sample.get('rgb_context', None)
    org_images = sample.get('rgb_original', None)
    org_contexts = sample.get('rgb_context_original', None)    
    ego_poses = sample.get('ego_pose', None)

    n_cam, _, w, h = aug_images.shape

    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    sample['K'] = resized_K
    
    if 'extrinsics' in sample:
        sample['c2e_extr'] = sample['extrinsics']
    
    sample[('color_aug', 0)] = aug_images
    
    if org_images is not None:
        sample[('color_org', 0)] = org_images
    
    # ego_poses now contains absolute ego-to-world transformations for each camera
    # We store them directly without computing ego_T_ego here (will be computed later when we have both frames)
    if ego_poses is not None:
        sample['ego_pose'] = ego_poses  # Keep the absolute ego pose

    if contexts and aug_contexts is not None:
        for idx, frame in enumerate(contexts):
            sample[('color_aug', frame)] = aug_contexts[idx]
            if org_contexts is not None:
                sample[('color_org', frame)] = org_contexts[idx]

    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            del sample[key]

        for np_key in _NP_KEYS:
            if np_key in key:
                # Use float64 for timestamp to preserve precision (NuScenes timestamps are int64 microseconds)
                # Float32 loses precision for large timestamp values (~1.5e15)
                if np_key == 'timestamp':
                    sample[key] = torch.from_numpy(sample[key]).to(torch.float64)
                else:
                    sample[key] = torch.from_numpy(sample[key]).to(torch.float32)
    
    # Preserve vehicle_annotations without tensor conversion
    # as they contain mixed data types (strings, lists, objects)
    if 'vehicle_annotations' in sample:
        # Keep as-is, don't convert to tensor
        pass
    
    return sample



def to_tensor(x: Union[np.ndarray, List, Tuple]) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x


LIDAR_NUM_RINGS = 32
LIDAR_AZIMUTH_BINS = 3600
LIDAR_ELEVATIONS = np.linspace(-30, 10, LIDAR_NUM_RINGS)
LIDAR_AZIMUTH_RESOLUTION = 360.0 / LIDAR_AZIMUTH_BINS


def build_lidar_range_image(lidar_points, lidar_to_ego, ego_to_world, lidar_timestamp):
    xyz = lidar_points[:, :3]
    intensity = lidar_points[:, 3] / 255.0
    ring = lidar_points[:, 4].astype(int)

    ones = np.ones((xyz.shape[0], 1))
    xyz_homo = np.concatenate([xyz, ones], axis=1)
    xyz_ego = (lidar_to_ego @ xyz_homo.T).T

    depth = np.linalg.norm(xyz, axis=1)
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    elevation = np.degrees(np.arcsin(xyz[:, 2] / (depth + 1e-8)))

    H, W = LIDAR_NUM_RINGS, LIDAR_AZIMUTH_BINS

    gt_depth = np.zeros((H, W), dtype=np.float32)
    gt_intensity = np.zeros((H, W), dtype=np.float32)
    gt_ray_drop = np.zeros((H, W), dtype=np.float32)

    for r in range(LIDAR_NUM_RINGS):
        mask = ring == r
        if not mask.any():
            continue
        pts_az = azimuth[mask] % 360
        pts_d = depth[mask]
        pts_int = intensity[mask]
        col = (pts_az / LIDAR_AZIMUTH_RESOLUTION).astype(int) % W
        for ci in range(W):
            bin_mask = col == ci
            if bin_mask.any():
                nearest = np.argmin(pts_d[bin_mask])
                gt_depth[r, ci] = pts_d[bin_mask][nearest]
                gt_intensity[r, ci] = pts_int[bin_mask][nearest]
                gt_ray_drop[r, ci] = 1.0

    az_grid, el_grid = np.meshgrid(
        np.linspace(0, 360, W, endpoint=False),
        LIDAR_ELEVATIONS,
    )
    depth_init = gt_depth.copy()
    depth_init[depth_init == 0] = 50.0
    raster_pts = np.stack([
        az_grid, el_grid, depth_init, np.zeros_like(az_grid)
    ], axis=-1)

    tile_height = 8
    el_boundaries = np.linspace(-30, 10, LIDAR_NUM_RINGS // tile_height + 1)

    return (
        raster_pts[None],
        gt_depth[None, ..., None],
        gt_intensity[None, ..., None],
        gt_ray_drop[None, ..., None],
        el_boundaries,
    )


def to_float_tensor(d):
    if isinstance(d, dict):
        return {k: to_float_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_float_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.float()
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d