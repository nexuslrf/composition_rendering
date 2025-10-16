import bpy
import numpy as np
import random
import os
import json
import copy
from mathutils import Vector

from utils import blender_utils, render_utils, image_utils
import torch
from physics import rigid_body_utils as rb


def _sample_velocity(downward_bias=0.7, speed_range=(0.0, 3.0)):
    speed = random.uniform(*speed_range)
    # Biased towards -Z with lateral noise
    dir_vec = np.array([random.uniform(-1, 1), random.uniform(-1, 1), -1.0])
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
    down = np.array([0.0, 0.0, -1.0])
    blended = downward_bias * down + (1.0 - downward_bias) * dir_vec
    blended = blended / (np.linalg.norm(blended) + 1e-8)
    return (blended * speed).tolist()


def _sample_angular_speed(range_deg_s=(0.0, 30.0)):
    # Random axis with random speed mapped to radians
    deg_s = random.uniform(*range_deg_s)
    rad_s = np.deg2rad(deg_s)
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return (axis * rad_s).tolist()


def run(mesh_list, mesh_meta, envlight_path_list, shortname, prefix, FLAGS):
    # Camera setup (randomized similar to vtran_obj / rotat_obj)
    cam_radius = FLAGS.radius_range[0] + np.random.uniform() * (FLAGS.radius_range[1] - FLAGS.radius_range[0])
    fovx = np.deg2rad(FLAGS.fov_range[0] + np.random.uniform() * (FLAGS.fov_range[1] - FLAGS.fov_range[0]))
    azimuth = np.random.uniform(*FLAGS.cam_phi_range)
    elevation = np.random.uniform(*FLAGS.cam_theta_range)
    if FLAGS.cam_t_range is not None:
        t = np.random.uniform(*FLAGS.cam_t_range, size=[3])
    else:
        t = np.zeros(3)

    # Optional varying radius like in render_scene
    cam_radius_list = None
    if getattr(FLAGS, 'varying_radius', False):
        if random.random() < 0.3:
            roll_step = np.random.uniform(-np.pi/2, np.pi/2)
            cam_radius_list = FLAGS.radius_range[0] + (FLAGS.radius_range[1] - FLAGS.radius_range[0]) * \
                (1 + np.sin(np.linspace(0, 2*np.pi, FLAGS.num_frames, endpoint=False) + roll_step)) / 2

    cam_list = []
    for it in range(FLAGS.num_frames):
        radius_frame = cam_radius if cam_radius_list is None else cam_radius_list[it]
        cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, radius_frame)
        cam_list.append(cam_matrix)

    blender_utils.setup_realtime_camera_update(cam_list, cam_mode='MATRIX')
    blender_utils.setup_camera_settings(
        resolution_x=FLAGS.resolution[1], resolution_y=FLAGS.resolution[0], fov_rad=fovx,
    )

    # Physics world
    rb.ensure_rigidbody_world(
        gravity=tuple(FLAGS.physics.get('gravity', [0.0, 0.0, -9.81])),
        steps_per_second=int(FLAGS.physics.get('steps_per_second', 240)),
        substeps_per_frame=int(FLAGS.physics.get('substeps_per_frame', 5)),
        solver_iterations=int(FLAGS.physics.get('solver_iterations', 10)),
        split_impulse=bool(FLAGS.physics.get('split_impulse', True)),
        cache_frames=int(FLAGS.physics.get('cache_frames', 250)),
    )
    # Ground and optional walls
    # Prefer using the existing placement plane (first mesh in mesh_list) to preserve its material/texture.
    if len(mesh_list) > 0 and FLAGS.physics.get('set_rigidbody_plane', True):
        try:
            rb.add_passive_rigidbody(mesh_list[0], collision_shape='MESH', friction=0.8, restitution=0.0)
        except Exception:
            ground_size = float(FLAGS.environment.get('ground_size', 8.0))
            rb.add_passive_ground(size=ground_size, location=FLAGS.placement_plane_offset, transparent=True)
    else:
        ground_size = float(FLAGS.environment.get('ground_size', 8.0))
        rb.add_passive_ground(size=ground_size, location=FLAGS.placement_plane_offset, transparent=True)
    if FLAGS.environment.get('walls', {}).get('enabled', False):
        size_xy = FLAGS.environment['walls'].get('size', [6.0, 6.0])
        height = float(FLAGS.environment['walls'].get('height', 2.0))
        x, y = size_xy
        rb.add_wall(size_x=x, size_y=0.2, height=height, location=(0.0, -y/2.0, height/2.0), name='WallSouth')
        rb.add_wall(size_x=x, size_y=0.2, height=height, location=(0.0,  y/2.0, height/2.0), name='WallNorth')
        rb.add_wall(size_x=0.2, size_y=y, height=height, location=(-x/2.0, 0.0, height/2.0), name='WallWest')
        rb.add_wall(size_x=0.2, size_y=y, height=height, location=( x/2.0, 0.0, height/2.0), name='WallEast')

    # Spawn region
    region = FLAGS.spawn.get('region', {'center': [0.0, 0.0, 1.5], 'size': [1.0, 1.0, 0.4]})
    center = np.array(region.get('center', [0.0, 0.0, 1.5]), dtype=float)
    size = np.array(region.get('size', [1.0, 1.0, 0.4]), dtype=float)

    # Create active bodies for objects (skip plane if present at index 0)
    physics_cfg = FLAGS.physics
    for idx in range(1, len(mesh_list)):
        rb.add_active_rigidbody(
            mesh_list[idx],
            mass=float(physics_cfg.get('mass', 1.0)),
            friction=float(random.uniform(*physics_cfg.get('friction_range', [0.3, 0.9]))),
            restitution=float(random.uniform(*physics_cfg.get('restitution_range', [0.2, 0.8]))),
            collision_shape=str(physics_cfg.get('collision_shape', 'CONVEX_HULL')),
            collision_margin=float(physics_cfg.get('collision_margin', 0.001)),
            use_deactivation=bool(physics_cfg.get('use_deactivation', True)),
        )

        # Sample spawn position and initial velocities
        if getattr(mesh_list[idx], 'empty', None) is not None:
            loc = center + (np.random.rand(3) - 0.5) * size
            # Ensure above ground
            loc[2] = max(loc[2], 0.6)
            mesh_list[idx].empty.location = Vector(loc.tolist())
        lin_v = _sample_velocity(
            downward_bias=float(FLAGS.initial_motion.get('downward_bias', 0.7)),
            speed_range=tuple(FLAGS.initial_motion.get('speed_range', [0.0, 3.0]))
        )
        ang_v = _sample_angular_speed(tuple(FLAGS.initial_motion.get('angular_speed_range', [0.0, 30.0])))
        rb.set_initial_velocity(mesh_list[idx], linear=lin_v, angular=ang_v)

    # After keyframing the initial conditions, ensure frame range
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FLAGS.num_frames

    # Bake physics once before rendering under multiple lights
    try:
        rb.bake_rigidbody_cache(frame_start=1, frame_end=FLAGS.num_frames)
    except Exception:
        for f in range(1, FLAGS.num_frames + 1):
            scene.frame_set(f)

    # Precompute envmap helper data for dumping
    dump_format = FLAGS.dump_format
    vec = None
    if FLAGS.dump_envmap:
        vec = render_utils.latlong_vec(FLAGS.resolution)

    ori_shortname = shortname
    blender_utils.setup_cycles_rendering(samples=FLAGS.spp, use_denoise=FLAGS.use_denoise, transparent_bg=FLAGS.transparent_bg)

    for lgt_i in range(FLAGS.num_lighting):
        # Prefix and folder handling similar to render_scene
        prefix = f'{lgt_i:04d}.'
        shortname = ori_shortname
        if getattr(FLAGS, 'prefix_in_folder', False):
            shortname = f'{ori_shortname}.{lgt_i:04d}'
            prefix = ''
            os.makedirs(os.path.join(FLAGS.out_dir, shortname), exist_ok=True)

        save_folder = f"{FLAGS.out_dir}/{shortname}"

        # Envmap augmentation
        if getattr(FLAGS, 'envlight_sample_weight', None) is not None:
            envlight_path = np.random.choice(envlight_path_list, p=FLAGS.envlight_sample_weight)
        else:
            envlight_path = random.choice(envlight_path_list)
        envmap_strength = np.random.uniform(*FLAGS.random_env_scale) if getattr(FLAGS, 'random_env_scale', None) is not None else FLAGS.env_scale
        envmap_flip = False
        if getattr(FLAGS, 'random_env_flip', False) and (random.random() > 0.5):
            envmap_flip = True
        if getattr(FLAGS, 'random_env_rotation', False):
            envmap_rotation_y = random.uniform(0, 2*np.pi)
        else:
            envmap_rotation_y = 0.0

        # Optional envmap dump (pass-level overview)
        cubemap = None
        if FLAGS.dump_envmap:
            latlong_img = image_utils.read_img(envlight_path)
            latlong_img = latlong_img * envmap_strength
            latlong_img = np.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0)
            latlong_img = np.clip(latlong_img, 0.0, 65504.0)
            latlong_img = torch.tensor(latlong_img, dtype=torch.float32)
            if envmap_flip:
                latlong_img = latlong_img.flip(1)
            cubemap = render_utils.latlong_to_cubemap_torch(latlong_img, [512, 512])
            env_proj = render_utils.cubemap_sample_torch(cubemap, -vec)
            env_proj = env_proj.flip(0).flip(1)
            env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
            env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_ldr.{dump_format}'), env_ev0)
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_log.{dump_format}'), env_log)

        # Set envmap in Blender for actual rendering
        blender_utils.set_envmap_texture(envlight_path, envmap_rotation_y, envmap_strength, envmap_flip)

        # Setup feature passes only for first lighting to save time
        if lgt_i == 0:
            blender_utils.setup_render_passes(['normal', 'depth', 'diffcol', 'object', 'material'])
            blender_utils.setup_compositor_nodes(output_dir=save_folder, passes=['normal', 'depth'], suffix=f'.{0:04d}')
            blender_utils.render_albedo_and_material(output_dir=save_folder, passes=['albedo', 'orm'], suffix=f'.{0:04d}')

        # Per-frame env projections if requested
        if FLAGS.dump_envmap and cubemap is not None:
            y_rot = render_utils.rotate_y(envmap_rotation_y, device=latlong_img.device)
            for it in range(FLAGS.num_frames):
                c2w = render_utils.convert_cam_mat_blender_to_dr(cam_list[it])
                c2w = torch.tensor(c2w, dtype=torch.float32, device=latlong_img.device)
                vec_cam = vec.reshape(-1, 3) @ c2w[:3, :3].T
                vec_query = (vec_cam @ y_rot[:3, :3].T).reshape(1, *FLAGS.resolution, 3)
                env_frame = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                env_frame = env_frame.flip(0).flip(1)
                env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_frame, max_point=16).clip(0, 1)).cpu().numpy()
                env_log = render_utils.rgb_to_srgb(torch.log1p(env_frame) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_ldr.{dump_format}'), env_ev0)
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_log.{dump_format}'), env_log)

        # Render RGB frames for this lighting setup
        blender_utils.render_all_frames(output_dir=save_folder, num_frames=FLAGS.num_frames, suffix=f'rgb.{lgt_i:04d}')

        # Dump meta (per lighting pass, prefixed)
        meta_dict = {
            'camera_angle_x': fovx,
            'cam_radius': FLAGS.radius_range[0],
            'envmap': os.path.basename(envlight_path),
        }
        meta_frames = []
        for it in range(FLAGS.num_frames):
            meta_frames.append({
                'transform_matrix': cam_list[it].tolist(),
                'elevation': float(elevation),
                'azimuth': float(azimuth),
                'envmap_rot': float(envmap_rotation_y),
                'envmap_strength': float(envmap_strength),
                'envmap_flip': bool(envmap_flip),
            })
        meta_dict['frames'] = meta_frames
        meta_dict['file_path'] = shortname
        with open(os.path.join(FLAGS.out_dir, shortname, f'{prefix}meta.json'), 'w') as f:
            _meta_dict = copy.deepcopy(meta_dict)
            _meta_dict['mesh_list'] = mesh_meta
            json.dump(_meta_dict, f, indent=4)


