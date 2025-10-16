### Drop Physics Mode

<div align="center">
  <img src="teaser/drop_phy.gif" alt=""  width="60%" />
</div>

Physically drops scene objects onto a ground plane using Blender's rigid body simulation, then renders a short sequence. Use this when you want realistic contacts, bounces, and frictional settling instead of purely kinematic motion.

---

### Quick start

Run the main generator with the provided config that sets `video_mode: drop_phy`:

```bash
python blender_datagen_compose.py \
  --config configs/render_drop_phy.yaml \
  out_dir=output/blender_drop_physics
```

Notes:
- You can override any config via dotlist CLI (e.g., `num_frames=120 physics.steps_per_second=240`).
- Output will be saved under `out_dir` inside a seed- and mode-specific subfolder (e.g., `drop_phy_sMMDDHH/000000/`).

---

### What this mode does

- Loads a textured ground plane (or creates a fallback plane) and converts it to a PASSIVE rigid body.
- Converts other meshes in the scene to ACTIVE rigid bodies with randomized friction and restitution.
- Samples initial linear velocity with a downward bias and random angular velocity for each object.
- Simulates for `num_frames` while rendering RGB and auxiliary passes (normal, depth, albedo, roughness, metallic) for the first lighting setup.
- Writes per-scene metadata (`meta.json`) with camera, environment, and mesh info.

---

### Configuration highlights (drop physics specific)

These sections directly influence the physics behavior in `drop_phy` mode:

```yaml
# Where objects spawn above the plane
spawn:
  region:
    center: [0.0, 0.0, 1.5]  # XYZ center of spawn box (meters)
    size:   [1.0, 1.0, 0.6]  # box side lengths; Z controls initial height span

# Initial motion sampling for each active body
initial_motion:
  speed_range: [0.0, 3.0]       # linear speed magnitude (m/s)
  downward_bias: 0.7            # 1.0 = fully downward; 0.0 = uniform directions
  angular_speed_range: [0.0, 30.0]  # deg/s, random axis

# Physics world and rigid body properties
physics:
  gravity: [0.0, 0.0, -9.81]
  steps_per_second: 240         # sim solver rate
  substeps_per_frame: 5         # extra substeps per rendered frame
  solver_iterations: 20         # constraint iterations
  split_impulse: true           # reduce interpenetration/bounce artifacts
  cache_frames: 250             # min cache length; sim cache covers [1, max(num_frames, cache_frames)]
  collision_shape: CONVEX_HULL  # for active bodies; robust default
  collision_margin: 0.002       # meters
  restitution_range: [0.2, 0.8] # bounciness randomized per object
  friction_range: [0.3, 0.9]    # friction randomized per object
  use_deactivation: true        # allow sleeping when settled
  mass: 1.0                     # mass assigned to each active body

# Scene containment and ground size
environment:
  ground_size: 10.0
  walls:
    enabled: true               # add invisible walls around the ground
    size: [6.0, 6.0]            # XY span of the enclosure
    height: 2.0                 # wall height
```

Also relevant general keys (not specific to physics but used here):
- `num_frames`: simulation/render length.
- `resolution`, `spp`, `use_denoise`, `transparent_bg`: render quality and format.
- `envlight`, `env_scale`, `random_env_rotation`, `random_env_flip`, `random_env_scale`: HDRI lighting.
- `radius_range`, `fov_range`: camera setup (static in this mode).

---

### Tips

- Increase `steps_per_second` or `substeps_per_frame` for stability when objects are small or fast.
- Use `CONVEX_HULL` for robust collisions; `MESH` can be used for high-fidelity but is costlier and less stable.
- If you notice a small visible gap between objects and the ground, it's typically due to the collision margin on active bodies (CONVEX_HULL adds a thin shell). Lower `physics.collision_margin` (e.g., 0.0005–0.0) or use `MESH` for higher-fidelity contacts at the cost of stability.
- Enable `environment.walls.enabled` to keep fast objects from sliding off the plane.
- If objects jitter after settling, try higher `solver_iterations` and enable `use_deactivation`.
 - `physics.cache_frames` ensures the rigid body point cache is long enough even if `num_frames` is short. Set higher if you want the sim to continue caching beyond the render range (useful when scrubbing or when physics needs extra settling time).

---

### Example config

See `configs/render_drop_phy.yaml` for a complete, ready-to-run example covering all keys above.


