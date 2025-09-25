#!/usr/bin/env python3
"""
Blender Data Generation Script for Composition Rendering
Reimplements datagen_compose.py functionality using Blender's Python API
"""

import bpy
import bmesh
import math
import random
import json
import os
import shutil
import sys
import logging
import time
import argparse
import glob
import numpy as np
import copy
import torch
import imageio.v3 as iio
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from mathutils import Vector, Matrix, Quaternion
from pathlib import Path
from shapely.geometry import Polygon
from utils import blender_utils, render_utils, image_utils
import matplotlib.pyplot as plt

DEPTH_MAX = 1000.0

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s - %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)

# Create a logger
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def check_msh_bbox(msh):
    vmin, vmax = msh.aabb
    bbox = (vmax - vmin) # [x, y, z]
    hori_ratio = bbox[0] / bbox[2]
    if hori_ratio > 6 or hori_ratio < 1/6:
        return False
    else:
        vert_ratio = (bbox[1] / bbox[0], bbox[1] / bbox[2])
        if max(vert_ratio) < 0.15: # too flat
            return False
        if min(vert_ratio) > 8: # too tall
            return False
    return True

def post_process_rendering(output_dir, feature_fmt='jpg'):
    blender_passes = ['rgb', 'normal', 'depth', 'albedo', 'orm']
    mask = 0
    for p in blender_passes:
        img_list = sorted(glob.glob(os.path.join(output_dir, f'{p}.*')))
        if p == 'normal' and len(img_list) > 0:
            meta_file = json.load(open(os.path.join(output_dir, f'0000.meta.json')))
            meta_frames = meta_file['frames']
        for img_path in img_list:
            img_basename = os.path.basename(img_path)
            img_name_part = img_basename.split('.')
            pidx, fidx, img_fmt = int(img_name_part[1]), int(img_name_part[2]) - 1, img_name_part[3]
            img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{img_fmt}'
            if p == 'normal':
                w_normal = image_utils.read_normal_exr(img_path)[..., :3] # [H, W, 3]
                mask = (w_normal == 0).all(axis=-1, keepdims=True) # [H, W, 1]
                bg_normal = np.array([0, 0, 1])
                c2w = np.array(meta_frames[fidx]['transform_matrix'])
                w2c_rot = np.linalg.inv(c2w[:3, :3])
                s_normal = w_normal @ w2c_rot.T
                s_normal = s_normal * (1-mask) + mask * bg_normal
                s_normal = (s_normal + 1) * 0.5
                img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{feature_fmt}'
                image_utils.save_image(os.path.join(output_dir, img_new_name), s_normal)
                # remove the original normal
                os.remove(img_path)
            elif p == 'depth':
                depth = image_utils.read_depth_exr(img_path)
                iio.imwrite(os.path.join(output_dir, img_new_name), depth, plugin='opencv')
                os.remove(img_path)
            elif p == 'albedo':
                albedo = image_utils.read_img(img_path)
                img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{feature_fmt}'
                # NOTE: albedo is in sRGB!!!
                albedo = render_utils.rgb_to_srgb(albedo)
                image_utils.save_image(os.path.join(output_dir, img_new_name), albedo)
                os.remove(img_path)
            elif p == 'orm':
                orm = image_utils.read_img(img_path)
                roughness, metallic = orm[..., 1:2], orm[..., 2:3]
                for key, value, bg_color in zip(['roughness', 'metallic'], [roughness, metallic], [0.5, 0]):
                    img_new_name = f'{pidx:04d}.{fidx:04d}.{key}.{feature_fmt}'
                    value = value * (1-mask) + mask * bg_color
                    image_utils.save_image(os.path.join(output_dir, img_new_name), value[..., 0])
                os.remove(img_path)
            else:
                # os.rename(img_path, os.path.join(output_dir, img_new_name))
                shutil.move(img_path, os.path.join(output_dir, img_new_name))

def render_scene(
    mesh_list, mesh_meta, envlight_path_list, shortname, prefix, FLAGS
):
    cam_radius = FLAGS.radius_range[0] + np.random.uniform() * (FLAGS.radius_range[1] - FLAGS.radius_range[0])

    fovx = np.deg2rad(FLAGS.fov_range[0]+np.random.uniform()*(FLAGS.fov_range[1] - FLAGS.fov_range[0]))
    fovx_list = None
    azimuth = np.random.uniform(*FLAGS.cam_phi_range)
    elevation = np.random.uniform(*FLAGS.cam_theta_range)
    if FLAGS.cam_t_range is not None:
        t = np.random.uniform(*FLAGS.cam_t_range, size=[3])
    else:
        t = np.zeros(3)
    
    cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, cam_radius)

    num_frames = FLAGS.num_frames
    if FLAGS.video_mode == 'orbit_cam':
        azimuth_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        elevation_offset = np.zeros(num_frames)
    elif FLAGS.video_mode == 'oscil_cam':
        phi_center = sum(FLAGS.cam_phi_range) / 2
        phi_ratio = 0.6
        phi_range = (FLAGS.cam_phi_range[1] - FLAGS.cam_phi_range[0]) / 2
        phi_range_clip = phi_range * phi_ratio
        azimuth = np.random.uniform(phi_center - phi_range_clip, phi_center + phi_range_clip)
        cone_angle = np.random.uniform(0.2 * (1-phi_ratio) * phi_range, phi_range - np.abs(phi_center - azimuth))
        # azimuth = np.random.uniform(*FLAGS.cam_phi_range)
        azimuth_offset = np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * cone_angle
        elevation_offset = np.cos(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * cone_angle
    elif FLAGS.video_mode == 'dolly_cam':
        # start_fov
        fovx_fix = np.deg2rad(45)
        radius_fix = 2.5
        fovx_perturb = np.random.uniform(-10, 10, size=2)
        fovx_motion = np.linspace(
            max(10, FLAGS.fov_range[0] + fovx_perturb[0]),
            FLAGS.fov_range[1] + fovx_perturb[1],
            num_frames, endpoint=True
        )
        if random.random() < 0.5:
            fovx_motion = fovx_motion[::-1]
        fovx_list = []
    elif FLAGS.video_mode == 'orbit_lgt':
        env_rot_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    elif FLAGS.video_mode == 'rotat_obj':
        obj_rot_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        mesh_id = [1]
        if mesh_meta is not None and random.random() > 0.5:
            if len(mesh_list) > 2 and 'metallic' not in mesh_meta[2]:
                mesh_id.append(2)
    elif FLAGS.video_mode == 'vtran_obj':
        num_obj = len(mesh_list) - 1
        drop_id = np.random.permutation(num_obj) + 1
        drop_id = drop_id.tolist()
        drop_id = drop_id[:1] if random.random() < 0.5 else drop_id[:2]
        if 1 not in drop_id and random.random() < 0.8:
            drop_id = drop_id + [1]
        num_drop = len(drop_id)
        drop_prev = [0] * num_drop
        drop_range = [0.5, 1.5]
        drop_list = np.random.uniform(*drop_range, size=[num_drop])
        drop_offset_list = []
        bounce = random.random() < 0.5 # always bounce 1/3 of the height
        if bounce:
            bounce_nframes = num_frames // 3 + random.randint(-num_frames//6, num_frames//6)
        else:
            bounce_nframes = 0
        for drop in drop_list:
            drop_offset = np.linspace(drop, 0, num_frames - bounce_nframes, endpoint=True) # the last is 0
            if bounce:
                bounce_factor = random.uniform(0.33, 0.8)
                bounce_offset = np.linspace(drop * bounce_factor, 0, bounce_nframes, endpoint=False)[::-1] # the last might not be 0
                drop_offset = np.concatenate([drop_offset, bounce_offset])
            drop_offset_list.append(drop_offset)

    cam_radius_list = None
    if FLAGS.varying_radius:
        if random.random() < 0.3:
            # sin wave 
            # random roll
            roll_step = np.random.uniform(-np.pi/2, np.pi/2)
            cam_radius_list = FLAGS.radius_range[0] + (FLAGS.radius_range[1] - FLAGS.radius_range[0]) * \
                (1 + np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False) + roll_step )) / 2

    azimuth_0, elevation_0 = azimuth, elevation
    cubemap, vec, vec_ref, latlong_img = None, None, None, None
    if FLAGS.dump_envmap: # TODO:
        vec = render_utils.latlong_vec(FLAGS.resolution)
        vec_ball, mask = render_utils.get_ideal_normal_ball(FLAGS.resolution[0], flip_x=False)
        vec_ref = render_utils.get_ref_vector(vec_ball, np.array([0,0, 1]))
        vec_ref = vec_ref.float() #.to('cuda')
    
    ori_shortname = shortname
    skip_features = False
    dump_format = FLAGS.dump_format # TODO:
    for lgt_i in range(FLAGS.num_lighting):
        prefix = f'{lgt_i:04d}.'
        if FLAGS.prefix_in_folder:
            shortname = f'{ori_shortname}.{lgt_i:04d}'
            prefix = ''
            os.makedirs(os.path.join(FLAGS.out_dir, shortname), exist_ok=True)
            
        skip_features = lgt_i > 0
        if FLAGS.analytical_sky:
            raise NotImplementedError('Not supported yet')
        else:
            envlight_path = np.random.choice(envlight_path_list, p=FLAGS.envlight_sample_weight)
        
        envmap_strength = np.random.uniform(*FLAGS.random_env_scale) if FLAGS.random_env_scale is not None else FLAGS.env_scale
        envmap_flip = False
        if FLAGS.random_env_flip:
            if random.random() > 0.5:
                envmap_flip = True

        if FLAGS.random_env_rotation:
            envmap_rotation_y = random.uniform(0, 2*np.pi)
        else:
            envmap_rotation_y = 0
        envmap_rotation_y_0 = envmap_rotation_y

        if FLAGS.dump_envmap:
            latlong_img = image_utils.read_img(envlight_path)
            latlong_img = latlong_img * envmap_strength
            latlong_img = np.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0) 
            latlong_img = np.clip(latlong_img, 0.0, 65504.0)
            latlong_img = torch.tensor(latlong_img, dtype=torch.float32)
            if envmap_flip:
                latlong_img = latlong_img.flip(1)

            # TODO: rotate
            cubemap = render_utils.latlong_to_cubemap_torch(latlong_img, [512, 512])
            env_proj = render_utils.cubemap_sample_torch(cubemap, -vec)
            env_proj = env_proj.flip(0).flip(1)

            env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
            env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_ldr.{dump_format}'), env_ev0)
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_log.{dump_format}'), env_log)

            if FLAGS.dump_env_bg:
                intrinsic = render_utils.cam_intrinsics(fovx, FLAGS.resolution[1], FLAGS.resolution[0])
                env_uv = render_utils.uv_mesh(FLAGS.resolution[1], FLAGS.resolution[0])
                pos_cam = env_uv @ np.linalg.inv(intrinsic).T

        blender_utils.set_envmap_texture(envlight_path, envmap_rotation_y, envmap_strength, envmap_flip)    
        logger.info(f"EnvProbe {lgt_i}/{FLAGS.num_lighting}: {envlight_path}")

        meta_dict = {
            # 'tone_mapping': FLAGS.tonemap_type,
            # 'envmap': os.path.basename(envlight_path),
            'camera_angle_x': fovx, # fov along width
            'cam_radius': cam_radius,
        }
        if FLAGS.analytical_sky:
            raise NotImplementedError('Not supported yet')
        else:
            meta_dict['envmap'] = os.path.basename(envlight_path)
        meta_frames = []

        # =============================================================================================
        # Start rendering
        # =============================================================================================
        cam_list = []
        for it in range(num_frames):
            if FLAGS.video_mode in ['orbit_cam', 'oscil_cam']:
                azimuth = azimuth_0 + azimuth_offset[it]
                elevation = elevation_0 + elevation_offset[it]
                # to avoid camera or object flipping
                elevation = np.clip(elevation, 5*np.pi/180, 85*np.pi/180)
                cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, cam_radius)
                
            elif FLAGS.video_mode == 'orbit_lgt':
                envmap_rotation_y = envmap_rotation_y_0 + env_rot_offset[it]
                # blender_utils.rotate_envmap(envmap_rotation_y) # TODO:
                        
            elif FLAGS.video_mode == 'vtran_obj':
                pass
            elif FLAGS.video_mode == 'dolly_cam':
                fovx_frame = np.deg2rad(fovx_motion[it])
                radius_frame = radius_fix * np.tan(fovx_fix/2) / np.tan(fovx_frame/2)
                cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, radius_frame)
                fovx_list.append(fovx_frame)

            if FLAGS.varying_radius and cam_radius_list is not None and FLAGS.video_mode != 'dolly_cam':
                cam_radius = cam_radius_list[it]
                cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, cam_radius)
            
            cam_list.append(cam_matrix)

            if FLAGS.dump_envmap:
                c2w = render_utils.convert_cam_mat_blender_to_dr(cam_matrix)
                c2w = torch.tensor(c2w, dtype=torch.float32, device=vec.device)
                vec_cam = vec.reshape(-1, 3) @ c2w[:3, :3].T
                y_rot = render_utils.rotate_y(envmap_rotation_y, device=vec.device)
                vec_query = (vec_cam @ y_rot[:3, :3].T).reshape(1, *FLAGS.resolution, 3)
                env_proj = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                env_proj = env_proj.flip(0).flip(1)
                env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
                env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_ldr.{dump_format}'), env_ev0)
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_log.{dump_format}'), env_log)

                if FLAGS.dump_ball_env:
                    vec_ball = -vec_ref.reshape(-1, 3) @ c2w[:3, :3].T
                    vec_query = (vec_ball @ y_rot[:3, :3].T).reshape(1, FLAGS.resolution[0], FLAGS.resolution[0], 3)
                    env_proj = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                    env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
                    env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.ball_env_ldr.{dump_format}'), env_ev0)
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.ball_env_log.{dump_format}'), env_log)

                if FLAGS.dump_env_bg:
                    bg_dir = pos_cam @ c2w[:3, :3].T
                    bg_dir = (bg_dir @ y_rot[:3, :3].T)
                    bg_q_dir = -bg_dir.flip(1).contiguous().reshape(1, *FLAGS.resolution, 3)
                    bg_proj = render_utils.cubemap_sample_torch(cubemap, bg_q_dir)[0]
                    bg_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(bg_proj, max_point=16).clip(0, 1)).cpu().numpy()
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_bg.{dump_format}'), bg_ev0)



            meta_frame = {
                # camera attributes
                'transform_matrix': cam_matrix.tolist(), # standard blender c2w
                'elevation': elevation,
                'azimuth': azimuth,
                # envmap
                'envmap_rot': envmap_rotation_y,
                'envmap_strength': envmap_strength,
                'envmap_flip': envmap_flip,
            }
            if FLAGS.video_mode == 'rotat_obj':
                meta_frame['obj_rot'] = obj_rot_offset[it]
                meta_frame['obj_rot_id'] = mesh_id
            if FLAGS.video_mode == 'vtran_obj':
                meta_frame['drop_offset'] = [drop_offset_list[i][it] for i in range(num_drop)]
                meta_frame['drop_id'] = drop_id
            if FLAGS.video_mode == 'dolly_cam':
                meta_frame['fov'] = fovx_frame

            meta_frames.append(meta_frame)

        if not skip_features or lgt_i == 0: # only setup the camera update for the first lighting setup
            blender_utils.setup_realtime_camera_update(cam_list, cam_mode='MATRIX', fov_sequence=fovx_list)
            if FLAGS.video_mode == 'rotat_obj':
                for mi in mesh_id:
                    init_rot = mesh_list[mi].empty.rotation_euler[2]
                    mesh_list[mi].setup_realtime_update(num_frames, rotation=obj_rot_offset + init_rot)
            elif FLAGS.video_mode == 'vtran_obj':
                for di,mi in enumerate(drop_id):
                    init_loc = mesh_list[mi].empty.location
                    drop_offset_vec3 = np.zeros((num_frames, 3))
                    drop_offset_vec3[:, 2] = drop_offset_list[di]
                    drop_offset_vec3 += init_loc
                    mesh_list[mi].setup_realtime_update(num_frames, translation=drop_offset_vec3)
        
        if FLAGS.video_mode == 'orbit_lgt': # on orbit lgt needs reset.
            envmap_rotation_y_list = envmap_rotation_y_0 + env_rot_offset
            blender_utils.setup_realtime_envmap_update(envmap_rotation_y_list.tolist())

        # set up the rendering
        save_folder = os.path.join(FLAGS.out_dir, shortname)
        blender_utils.setup_camera_settings(
            resolution_x=FLAGS.resolution[1], resolution_y=FLAGS.resolution[0], fov_rad=fovx,
        )
        blender_utils.setup_cycles_rendering(samples=FLAGS.spp, use_denoise=FLAGS.use_denoise, transparent_bg=FLAGS.transparent_bg)

        blender_passes = ['rgb']
        if not skip_features:
            blender_utils.setup_render_passes(['normal', 'depth', 'diffcol', 'object', 'material'])
            blender_utils.setup_compositor_nodes(output_dir=save_folder, 
                passes=['normal', 'depth'], suffix=f'.{0:04d}') # NOTE: ['rgb', 'diffcol'] is removed here
            blender_utils.render_albedo_and_material(output_dir=save_folder, passes=['albedo', 'orm'], suffix=f'.{0:04d}')
            blender_passes.extend(['normal', 'depth', 'albedo', 'orm'])
        # else:
        #     blender_utils.setup_compositor_nodes(output_dir=save_folder, passes=['rgb'], suffix=f'.{lgt_i:04d}')
        blender_utils.render_all_frames(output_dir=save_folder, num_frames=num_frames, suffix=f'rgb.{lgt_i:04d}')
        
        # dump the meta data
        meta_dict['frames'] = meta_frames
        meta_dict['file_path'] = shortname
        with open(os.path.join(FLAGS.out_dir, shortname, f'{prefix}meta.json'), 'w') as f:
            _meta_dict = copy.deepcopy(meta_dict)
            _meta_dict['mesh_list'] = mesh_meta
            json.dump(_meta_dict, f, indent=4)

class GLTFFileManger:
    def __init__(
        self, files, random_sample=True, rescale=True, 
        multi_sample_weight=None, check_bbox=False
    ):
        # just assume files is a list of list of files
        self.files = files
        self.files_list = []
        for f in files:
            self.files_list.extend(f)
        self.num_file_lists = len(files)

        self.num_files = len(self.files_list)
        self.random_sample = random_sample
        self.rescale = rescale
        self.multi_sample_weight = multi_sample_weight
        self.check_bbox = check_bbox
        if self.multi_sample_weight is not None:
            assert len(self.multi_sample_weight) == self.num_file_lists
            
    def __len__(self):
        return self.num_files

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self): # inifi-gen
        self.idx += 1
        sample_success = False
        obj_mesh = None
        while not sample_success:
            if self.random_sample:
                idx = np.random.choice(self.num_file_lists, p=self.multi_sample_weight)
                file = np.random.choice(self.files[idx])
            else:
                file = self.files_list[(self.idx-1) % self.num_files]
                sample_success = True
            try:
                logger.info(f'Processing: {file}')
                _obj_mesh = blender_utils.add_object_file(
                    file, with_empty=True, recenter=True, rescale=self.rescale
                )
                if self.check_bbox and not check_msh_bbox(obj_mesh):
                    _obj_mesh.clear_objects()
                    del _obj_mesh
                    raise Exception('Invalid bbox')
                else:
                    obj_mesh = _obj_mesh
                    sample_success = True
            except Exception as e:
                logger.info(f'---> Error: {e}, skipping file {file}')
                self.files[idx].remove(file)
        
        return {'mesh': obj_mesh, 'name': file}
    
def main():
    parser = argparse.ArgumentParser(description='difftracer')
    parser.add_argument('-n', '--num_frames', type=int, default=1)
    parser.add_argument('-o', '--out_dir', type=str, default=".")
    parser.add_argument('--base_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--obj_shards', type=int, default=1, help='Number of shards for object sampling')
    parser.add_argument('--obj_shard_id', type=int, default=0, help='Shard index for object sampling')
    # parser.add_argument('--num_rendering', type=int, default=400)
    parser.add_argument('--sample_shape_texture', action='store_true', help='Sample shape texture', default=False)

    FLAGS = parser.parse_args()
    FLAGS.cam_near_far = [0.1, 1000.0]
    FLAGS.resolution = [256, 256] # [height, width]
    FLAGS.baseshape_path = None
    FLAGS.envlight = "data/envmap/aerodynamics_workshop_512.hdr"
    FLAGS.env_scale = 1.0
    FLAGS.env_res = [256,512]
    FLAGS.probe_res           = 256         # Env map probe resolution

    FLAGS.spp                 = 8
    FLAGS.use_denoise         = 'OPTIX'
    FLAGS.transparent_bg      = True
    # FLAGS.radius              = 3.0
    FLAGS.radius_range        = [2.0, 4.0]
    FLAGS.varying_radius      = False
    FLAGS.num_files           = None
    FLAGS.random_env_rotation = True
    FLAGS.random_env_flip     = True
    FLAGS.random_env_scale    = None
    FLAGS.bg_color            = [1.0, 1.0, 1.0]
    FLAGS.bg_features         = [0.0, 0.0, 0.0]
    FLAGS.dump_env_bg         = False
    FLAGS.dump_alpha          = False
    FLAGS.ray_depth           = 1
    FLAGS.num_workers         = 0
    FLAGS.timeout             = -1
    FLAGS.fov_range           = [45.0, 45.0] # Min/max fov in degrees
    FLAGS.dump_features       = False
    FLAGS.num_rendering       = 10
    FLAGS.num_lighting        = 1
    FLAGS.cam_phi_range       = [0, 360]
    FLAGS.cam_theta_range     = [0, 90]
    FLAGS.cam_t_range         = [0, 0]
    FLAGS.dump_shading        = False
    FLAGS.dump_splitsum       = False
    FLAGS.dump_envmap         = False
    FLAGS.dump_ball_env       = False
    FLAGS.dump_irradiance     = False
    FLAGS.dump_format         = 'jpg'
    FLAGS.dump_complete       = False
    FLAGS.dump_blend          = False
    FLAGS.dump_placement      = False
    FLAGS.video_mode          = 'orbit_cam' # 'orbit_cam', 'oscil_cam', 'orbit_lgt'
    # Sampling config
    FLAGS.glbs_check_bbox          = False
    FLAGS.glbs_multi_sample_weight = None
    FLAGS.glbs_rescale             = True
    FLAGS.glbs_random_sample       = True
    FLAGS.glbs_per_scene           = 2
    FLAGS.glbs_scale_range         = [1.0, 1.5] # [min, max]
    FLAGS.glbs_rotation_range      = [-45, 45]
    FLAGS.glbs_placement_bbox      = [-1.0, -1.0, 1.0, 1.0] # [min.x, min.y, max.x, max.y]
    FLAGS.shapes_per_scene         = 3
    FLAGS.shapes_scale_range       = [0.3, 0.8] # [min, max]
    FLAGS.shapes_rotation_range    = [-45, 45]
    FLAGS.shapes_placement_bbox    = [-1.5, -1.5, 1.5, 1.5] # [min.x, min.y, max.x, max.y]
   
    FLAGS.placement_centered       = False
    FLAGS.placement_bbox           = [-1.5, -1.5, 1.5, 1.5]
    FLAGS.placement_grid_res       = [40, 40]
    FLAGS.placement_bbox_scale     = 1.1
    FLAGS.placement_plane_offset   = [0, 0, -0.5]
    FLAGS.placement_plane_scale    = 10
    FLAGS.placement_plane          = "data/plane_basic/plane.glb"
    
    FLAGS.plane_sample_weight      = None
    FLAGS.envlight_sample_weight   = None
    FLAGS.texture_sample_weight    = None

    FLAGS.placement_plane_textures = None
    FLAGS.texture_match_string     = "*/*/textures"

    FLAGS.prefix_in_folder         = False

    ######### Unsupported flags #########
    # FLAGS.tonemap             = True
    # FLAGS.tonemap_type        = 'aces'
    # FLAGS.tonemap_func        = None
    # FLAGS.temporal_denoising       = False
    
    FLAGS.analytical_sky      = False
    FLAGS.use_objaverse       = False
    FLAGS.objaverse_selection = None
    #####################################

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]
    
    logger.info("Config / Flags:")
    logger.info("---------")
    for key in FLAGS.__dict__.keys():
        logger.info(f"{key}: {FLAGS.__dict__[key]}")
    logger.info("---------")

    # FLAGS.out_dir = 'out/' + FLAGS.out_dir
    sub_folder = f"s{FLAGS.seed:06d}" 
    # if FLAGS.iter > 1:
    sub_folder = f"{FLAGS.video_mode}_{sub_folder}"
    FLAGS.out_dir = os.path.join(FLAGS.out_dir, sub_folder)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    start_idx, round_idx = 0, 0
    prev_complete_file = None
    if FLAGS.dump_complete:
        # COMPLETE_ROUND_ITER
        round_offset = 100 # assume a job can be completed within 100 rounds of slurm submissions
        round_idx = 0
        complete_files = sorted(glob.glob("COMPLETE_*", root_dir=FLAGS.out_dir))
        if len(complete_files) > 0:
            complete_file = complete_files[-1]
            _, round_idx, start_idx = complete_file.split('_')
            start_idx, round_idx = int(start_idx) + 1, int(round_idx) + 1
        FLAGS.seed = FLAGS.seed * round_offset + round_idx
    
    set_seed(FLAGS.seed)

    FLAGS.cam_phi_range = [np.deg2rad(x) + np.pi for x in FLAGS.cam_phi_range]
    FLAGS.cam_theta_range = [np.deg2rad(x) for x in FLAGS.cam_theta_range]
    assert FLAGS.video_mode in ['orbit_cam', 'oscil_cam', 'orbit_lgt', 'none', 'rotat_obj', 'vtran_obj', 'dolly_cam']

    # Composition Sampling Related
    glbs_placement_vmin, glbs_placement_vmax = np.array(FLAGS.glbs_placement_bbox[:2]), np.array(FLAGS.glbs_placement_bbox[2:])
    shapes_placement_vmin, shapes_placement_vmax = np.array(FLAGS.shapes_placement_bbox[:2]), np.array(FLAGS.shapes_placement_bbox[2:])
    bbox_scale = 0.5 * FLAGS.placement_bbox_scale

    placement_vmin, placement_vmax = np.array(FLAGS.placement_bbox[:2]), np.array(FLAGS.placement_bbox[2:])
    placement_range = placement_vmax - placement_vmin
    placement_grid = np.zeros(FLAGS.placement_grid_res, dtype=np.int32)
    placement_bbox2grid = np.array(FLAGS.placement_grid_res) / placement_range
    placement_plane_offset = np.array(FLAGS.placement_plane_offset, dtype=np.float32)
    placement_plane_scale = np.array(FLAGS.placement_plane_scale, dtype=np.float32)
    placement_plane_path = FLAGS.placement_plane

    if placement_plane_path is None:
        placement_plane_path = []
    elif os.path.isdir(placement_plane_path):
        placement_plane_path = sorted(glob.glob(placement_plane_path + "/*.glb"))
    else:
        placement_plane_path = [placement_plane_path]

    num_planes = len(placement_plane_path)
    plane_sample_weight = np.ones(num_planes)
    if FLAGS.plane_sample_weight is not None:
        for i, plane in enumerate(placement_plane_path):
            for k, w in FLAGS.plane_sample_weight.items():
                if k in plane:
                    plane_sample_weight[i] = w
    FLAGS.plane_sample_weight = plane_sample_weight / plane_sample_weight.sum()

    def sample_glb(data_iter, obj_idx=0):
        target = next(data_iter) # centered, scaled
        ref_mesh = target['mesh']
        mesh_name = target['name']

        if ref_mesh is None:
            logger.info(f'Mesh {mesh_name} is None')
            return None
        num_materials = ref_mesh.get_num_materials()
        if num_materials == 0:
            logger.info(f'Mesh {mesh_name} has invalid materials {num_materials}')
            return None

        # Sample the object rotation
        smpl_rot_deg = np.random.uniform(*FLAGS.glbs_rotation_range)
        ref_mesh.apply_transform((0, 0, 0), rotation=np.deg2rad(smpl_rot_deg))
        # Sample the object scale
        smpl_scale = np.random.uniform(*FLAGS.glbs_scale_range)
        vmin, vmax = ref_mesh.aabb
        vmin, vmax = vmin * smpl_scale, vmax * smpl_scale
        mesh_center = np.array((vmax + vmin) * 0.5)
        cz = (mesh_center[2] - vmin[2])
        mesh_bounds = np.array(vmax - vmin)[:2]
        if FLAGS.video_mode == 'rotat_obj':
            # use square bbox
            mesh_bounds[:] = np.max(mesh_bounds)
        mesh_gbounds = mesh_bounds * placement_bbox2grid
        bx, by = int(np.ceil(mesh_gbounds[0] * bbox_scale)), int(np.ceil(mesh_gbounds[1] * bbox_scale))
        
        mesh_placement_vmin =  glbs_placement_vmin + mesh_bounds * 0.5
        mesh_placement_vmax =  glbs_placement_vmax - mesh_bounds * 0.5

        if not FLAGS.placement_centered:
            # Sample the object placement
            find_placement = False
            for t in range(5): # 8 tries
                # sample the center
                cx = np.random.uniform(mesh_placement_vmin[0], mesh_placement_vmax[0])
                cy = np.random.uniform(mesh_placement_vmin[1], mesh_placement_vmax[1])
                # convert to grid
                gx = round(((cx - placement_vmin[0]) * placement_bbox2grid[0]).item())
                gy = round(((cy - placement_vmin[1]) * placement_bbox2grid[1]).item())
                # check if the placement is valid
                x_lb, x_ub = max(gx-bx, 0), gx+bx
                y_lb, y_ub = max(gy-by, 0), gy+by
                if not placement_grid[x_lb:x_ub, y_lb:y_ub].any():
                    find_placement = True
                    placement_grid[x_lb:x_ub, y_lb:y_ub] += (obj_idx + 1)
                    break
                
            if not find_placement:
                logger.info(f'Cannot find valid placement for {mesh_name}')
                ref_mesh.clear_objects()
                del ref_mesh
                return None
        else:
            cx, cy = 0, 0
            cz = 0 if FLAGS.glbs_rescale else mesh_center[2]

        # logger.info('cx, cy, cz, smpl_scale', cx, cy, cz, smpl_scale)
        smpl_translation = np.array((cx, cy, cz))
        smpl_translation = smpl_translation - mesh_center + placement_plane_offset
        
        ref_mesh.apply_transform(smpl_translation, scale=smpl_scale)

        placement_meta = {
            'name': mesh_name,
            'translation': smpl_translation.tolist(),
            'rotation': smpl_rot_deg,
            'scale': smpl_scale,
            'num_materials': num_materials
        }

        return ref_mesh, placement_meta

    def sample_shape(shapes_files, obj_idx=0):
        target_file = np.random.choice(shapes_files)
        ref_mesh = blender_utils.add_object_file(target_file, with_empty=True, recenter=True, rescale=True)
        mesh_name = target_file

        # Sample the object rotation
        smpl_rot_deg = np.random.uniform(*FLAGS.shapes_rotation_range)
        ref_mesh.apply_transform((0, 0, 0), rotation=np.deg2rad(smpl_rot_deg))
        # Sample the object scale
        smpl_scale = np.random.uniform(*FLAGS.shapes_scale_range)
        vmin, vmax = ref_mesh.aabb
        vmin, vmax = vmin * smpl_scale, vmax * smpl_scale
        mesh_center = np.array((vmax + vmin) * 0.5)
        cz = (mesh_center[2] - vmin[2])
        mesh_bounds = np.array(vmax - vmin)[:2]
        mesh_gbounds = mesh_bounds * placement_bbox2grid
        bx, by = int(np.ceil(mesh_gbounds[0] * bbox_scale)), int(np.ceil(mesh_gbounds[1] * bbox_scale))
        mesh_placement_vmin =  shapes_placement_vmin + mesh_bounds * 0.5
        mesh_placement_vmax =  shapes_placement_vmax - mesh_bounds * 0.5

        # Sample the object placement
        find_placement = False
        for t in range(8): # 8 tries
            # sample the center
            cx = np.random.uniform(mesh_placement_vmin[0], mesh_placement_vmax[0])
            cy = np.random.uniform(mesh_placement_vmin[1], mesh_placement_vmax[1])
            # convert to grid
            gx = round(((cx - placement_vmin[0]) * placement_bbox2grid[0]).item())
            gy = round(((cy - placement_vmin[1]) * placement_bbox2grid[1]).item())
            # check if the placement is valid
            x_lb, x_ub = max(gx-bx, 0), gx+bx
            y_lb, y_ub = max(gy-by, 0), gy+by
            if not placement_grid[x_lb:x_ub, y_lb:y_ub].any():
                find_placement = True
                placement_grid[x_lb:x_ub, y_lb:y_ub] += (obj_idx + 1)
                break

        if not find_placement:
            logger.info(f'Cannot find valid placement for {mesh_name}')
            ref_mesh.clear_objects()
            del ref_mesh
            return None

        smpl_translation = np.array((cx, cy, cz))
        smpl_translation = smpl_translation - mesh_center + placement_plane_offset
        ref_mesh.apply_transform(smpl_translation, scale=smpl_scale)

        # Material sampling
        roughness_range = [0, 0.8]
        roughness = np.random.uniform(roughness_range[0], roughness_range[1]) ** 2
        metallic = np.abs(np.random.normal(0, 0.25))
        # 50% metallic (close to 1)
        if np.random.uniform() < 0.5:
            metallic = 1 - metallic
        metallic = 1.0 # NOTE: debug line
        if metallic > 0.5:
            base_color = np.random.randint(170, 255, size=3) / 255
        else:
            base_color = np.random.randint(30, 240, size=3) / 255

        logger.info(f'shape material: {base_color}, {roughness}, {metallic}')
        ref_mesh.set_principled_material(base_color, roughness, metallic)

        placement_meta = {
            'name': mesh_name,
            'translation': smpl_translation.tolist(),
            'rotation': smpl_rot_deg,
            'scale': smpl_scale,
            'roughness': roughness,
            'metallic': metallic,
            'base_color': base_color.tolist()
        }

        return ref_mesh, placement_meta

    # plane textures
    plane_textures_path = []
    if FLAGS.placement_plane_textures is not None:
        plane_textures_path = sorted(glob.glob(os.path.join(FLAGS.placement_plane_textures, FLAGS.texture_match_string)))
        num_plane_textures = len(plane_textures_path)
        texture_sample_weight = np.ones(num_plane_textures)
        if FLAGS.texture_sample_weight is not None:
            for i, texture in enumerate(plane_textures_path):
                for k, w in FLAGS.texture_sample_weight.items():
                    if k in texture:
                        texture_sample_weight[i] = w
        FLAGS.texture_sample_weight = texture_sample_weight / texture_sample_weight.sum()
        
    obj_files = [] 
    if not FLAGS.use_objaverse:
        if not os.path.isfile(FLAGS.base_path):
            obj_files.append(glob.glob(os.path.join(FLAGS.base_path, "*.glb")) + \
                glob.glob(os.path.join(FLAGS.base_path, "*.gltf")) + \
                glob.glob(os.path.join(FLAGS.base_path, "*.obj")))
        else:
            obj_files.append(np.loadtxt(FLAGS.base_path, dtype=str).tolist())
    else:
        raise NotImplementedError("Objaverse is not supported yet")

   
    # obj dataloader
    obj_dataloader = GLTFFileManger(
        obj_files, 
        random_sample=FLAGS.glbs_random_sample, 
        rescale=FLAGS.glbs_rescale, 
        multi_sample_weight=FLAGS.glbs_multi_sample_weight, 
        check_bbox=FLAGS.glbs_check_bbox
    )
    logger.info(f"Use {len(obj_dataloader)} glbs")

    # envlight 
    if not os.path.isfile(FLAGS.envlight):
        envlight_path_list = sorted(
            glob.glob(os.path.join(FLAGS.envlight, "*.exr"))  + \
            glob.glob(os.path.join(FLAGS.envlight, "*.hdr"))
        )
    elif FLAGS.envlight.endswith('.txt'):
        envlight_path_list = np.loadtxt(FLAGS.envlight, dtype=str).tolist()
    else:
        envlight_path_list = [FLAGS.envlight]

    num_envlights = len(envlight_path_list)
    envlight_sample_weight = np.ones(num_envlights)
    if FLAGS.envlight_sample_weight is not None:
        for i, envlight in enumerate(envlight_path_list):
            for k, w in FLAGS.envlight_sample_weight.items():
                if k in envlight:
                    envlight_sample_weight[i] = w
    FLAGS.envlight_sample_weight = envlight_sample_weight / envlight_sample_weight.sum()

    baseshape_files = []
    if FLAGS.baseshape_path is not None:
        if os.path.isdir(FLAGS.baseshape_path):
            baseshape_files = glob.glob(os.path.join(FLAGS.baseshape_path, "*.glb"))
        else:
            baseshape_files = [FLAGS.baseshape_path]
    
    iter_start_time = time.time()
    obj_iter = iter(obj_dataloader)
    for i in range(start_idx, FLAGS.num_rendering):
        logger.info(f"Rendering iteration {i}/{FLAGS.num_rendering}")
        name = f"{i:06d}"
        prefix = ''
        if FLAGS.num_frames > 1:
            prefix = f"{0:04d}."
        placement_grid = placement_grid * 0
        blender_utils.clear_scene()
        mesh_list = []
        mesh_meta = []

        # sample placement plane
        if num_planes > 0:
            plane_idx = np.random.choice(num_planes, size=1, p=FLAGS.plane_sample_weight)[0]
            # insert plane 
            placement_plane = blender_utils.add_object_file(placement_plane_path[plane_idx], with_empty=True, recenter=True, rescale=True)
            placement_plane.apply_transform((0, 0, 0), scale=placement_plane_scale)
            plane_vmin, plane_vmax = placement_plane.aabb
            vz = plane_vmin[2]
            vplane = np.array(placement_plane_offset) - np.array([0, 0, vz])
            placement_plane.apply_transform(vplane)
            # 
            plane_meta = {'name': os.path.basename(placement_plane_path[plane_idx])}
            # sample the plane texture...
            if num_plane_textures > 0:
                plane_tex_idx = np.random.choice(num_plane_textures, size=1, p=FLAGS.texture_sample_weight)[0]
                texture_scale = np.random.uniform(1.5, 2.5)
                placement_plane.apply_texture(plane_textures_path[plane_tex_idx], texture_scale)
                plane_meta['texture'] = plane_textures_path[plane_tex_idx]
                plane_meta['texture_scale'] = texture_scale
            mesh_meta.append(plane_meta)
            mesh_list.append(placement_plane)

        # add glbs
        for j in range(FLAGS.glbs_per_scene):
            target = sample_glb(obj_iter, obj_idx=j)
            if target is not None:
                ref_mesh, meta = target
                mesh_list.append(ref_mesh)
                mesh_meta.append(meta)


        num_glbs = len(mesh_list) - 1
        shape_list = []
        shape_meta = []
        for j in range(FLAGS.shapes_per_scene):
            target = sample_shape(baseshape_files, obj_idx=j+num_glbs)
            if target is not None:
                ref_mesh, meta = target
                shape_list.append(ref_mesh)

                if FLAGS.sample_shape_texture and num_plane_textures > 0:
                    if random.random() < 0.25:
                        plane_tex_idx = np.random.choice(num_plane_textures, size=1, p=FLAGS.texture_sample_weight)[0]
                        texture_scale = 1.0
                        ref_mesh.apply_texture(plane_textures_path[plane_tex_idx], texture_scale)
                        meta['texture'] = plane_textures_path[plane_tex_idx]
                        meta['texture_scale'] = texture_scale
                shape_meta.append(meta)

        if len(shape_list) > 0:
            mesh_list.extend(shape_list)
            mesh_meta.extend(shape_meta)
        else:
            logger.info("No shapes added")

        # Render multiple views and dump results
        if not FLAGS.prefix_in_folder:
            os.makedirs(os.path.join(FLAGS.out_dir, name), exist_ok=True)

        render_scene(mesh_list, mesh_meta, envlight_path_list, name, prefix, FLAGS)

        post_process_rendering(os.path.join(FLAGS.out_dir, name), feature_fmt=FLAGS.dump_format)

        if FLAGS.dump_blend:
            # save blender scene
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FLAGS.out_dir, name, f"scene.blend"))
        if FLAGS.dump_placement:
            # Plot placement grid with some color map
            fig = plt.figure()
            plt.imshow(placement_grid.T, cmap='tab20', vmin=0, vmax=placement_grid.max().item() + 1)
            # plt.axis('off')
            plt.savefig(os.path.join(FLAGS.out_dir, name, f"placement.png"))
            plt.close(fig)

    # clean up and safely exit blender
    blender_utils.clear_scene()
    bpy.ops.wm.quit_blender()
    # sys.exit(0)

if __name__ == "__main__":
    main()
