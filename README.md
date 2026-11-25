# ACG Rendering Track baseline

CPU build: `g++ -O3 -std=c++17 -x c++ render.cu -o render_cpu`.

```sh
python3 physics.py snapshots --seed 42 --fps 60 --frames 0 90
./render_cpu 2 1280 704 48 before.ppm 0 snapshots/before.txt
./render_cpu 2 1280 704 48 after.ppm 0 snapshots/after.txt
```

Pass additional uniformly spaced snapshots after the first snapshot to sample an exposure interval. The CPU backend does not require CUDA.

| Project Announcement §5.1 basic item | Implementation / evidence |
|---|---|
| Path tracing; diffuse and specular | Stages 1–2 in `scene.h`; stage 0 is diagnostic direct lighting only. |
| Custom scene | Orthographic overhead view; simplified analytical table, rails, and pockets; analytical spheres use the simulator's centers and radius. |
| Acceleration structure | Self-written stackless static BVH; balls use analytical intersections, with no tessellation. |
| Transmissive material | Glass sphere, dielectric reflection/refraction and Fresnel. |
| Original color texture | Ball-local procedural wood, veined marble, glazed ceramic, woven pattern and numbered enamel. |
| Importance sampling and Russian roulette | BSDF/PDF weighting and path termination in `trace`. |
| Motion blur and depth of field | Exposure samples from independent physics snapshots; orthographic thin lens (`ACG_PINHOLE` for control). |
| Alpha shadow | Perforated scene card with alpha visibility (`ACG_OPAQUE_MASK` for control). |
| Point light and area light | Point light diagnostic, rectangular emitter shading in stage 2. |
| Anti-aliasing | Pixel jitter (`ACG_NO_AA` for control). |

Implementation coverage is not a claim of course acceptance. The overhead hero frames are not standalone evidence for alpha shadows, transmission, or DOF. Targeted control renders should accompany the report. No GPU jobs or Classroom submission are needed to build this baseline.
