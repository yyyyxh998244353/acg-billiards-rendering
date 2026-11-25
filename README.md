# ACG Rendering Track baseline

CPU build: `g++ -O3 -std=c++17 -x c++ render.cu -o render_cpu`.

```sh
python3 physics.py snapshots --seed 42 --fps 60 --frames 0 90
./render_cpu 2 1280 704 48 before.ppm 0 snapshots/before.txt
./render_cpu 2 1280 704 48 after.ppm 0 snapshots/after.txt
```

Pass additional uniformly spaced snapshots after the first snapshot to sample an exposure interval. The CPU backend does not require CUDA.

For a replay, put one `snapshot.txt output.ppm` pair per line in a manifest (paths relative to the manifest; quote paths containing spaces):

```sh
ACG_BVH=refit ./render_cpu 2 1280 720 1 --sequence 0 replay.txt > timings.csv
```

To generate physics snapshots, render, and encode an animated WebP in one command (Python with Pillow):

```sh
python3 preview_replay.py ./render_cpu outputs/my-replay --width 128 --height 72 --last-frame 16
```

Use a fresh output directory. For cluster GPU runs, invoke with `--phase prepare` before allocation, `--phase render` inside the allocation, and `--phase encode` after release. Pass the same dimensions, frame count and frame rate to each phase. Only `all` and `encode` require Pillow; the script never submits Slurm jobs.

`ACG_BVH=scan|rebuild|refit` selects ball traversal (default `scan`). `ACG_INTERPOLATE=4` enables the first-hit felt point-shadow cache for single-pose frames. Build with `-DACG_ANALYTIC_SHADOW` to enable the separate rectangular-light estimator. Keep these options off for the reference. Frame CSV separates geometry preparation, cache, upload, render, readback, and file output.

To compile CUDA without reserving a GPU: `nvcc -O3 -std=c++17 -arch=sm_90 -Xptxas=-v render.cu -o render_gpu` (use `sm_80` for A100). Do not submit `run.slurm` under a one-second allocation limit: the script's guard is checked only after Slurm has allocated a device.

For the faster preview build, add `-DACG_MAX_HITS=2` and render at `--spp 6`. This trades higher-order indirect lighting for more camera samples. Keep the default 14 surface hits for reference images.

| Project Announcement §5.1 basic item | Implementation / evidence |
|---|---|
| Path tracing; diffuse and specular | Stages 1–2 in `scene.h`; stage 0 is diagnostic direct lighting only. |
| Custom scene | Orthographic overhead view; simplified analytical table, rails, and pockets; analytical spheres use the simulator's centers and radius. |
| Acceleration structure | Self-written stackless static BVH; balls use analytical intersections, with no tessellation. |
| Transmissive material | Build with `-DACG_GLASS_REFERENCE` for the separate glass-sphere render; dielectric reflection/refraction and Fresnel. The table render omits this prop. |
| Original color texture | Ball-local procedural wood, veined marble, glazed ceramic, woven pattern and numbered enamel. |
| Importance sampling and Russian roulette | BSDF/PDF weighting and path termination in `trace`. |
| Motion blur and depth of field | Exposure samples from independent physics snapshots; orthographic thin lens (`ACG_PINHOLE` for control). |
| Alpha shadow | Perforated scene card with alpha visibility (`ACG_OPAQUE_MASK` for control). |
| Point light and area light | Point light diagnostic, rectangular emitter shading in stage 2. |
| Anti-aliasing | Pixel jitter (`ACG_NO_AA` for control). |

Implementation coverage is not a claim of course acceptance. The overhead hero frames are not standalone evidence for alpha shadows, transmission, or DOF. Targeted control renders should accompany the report. No GPU jobs or Classroom submission are needed to build this baseline.
