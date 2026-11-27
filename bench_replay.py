"""Compare binaries built for the same scene and camera projection; never submits Slurm jobs."""
import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import subprocess


def run(binary, mode, spacing, spp, args, folder):
    folder.mkdir()
    manifest = folder / 'frames.txt'
    manifest.write_text(''.join(
        f'"{args.snapshots / f"frame_{i:06d}.txt"}" "{j:04d}.ppm"\n'
        for j, i in enumerate(args.frames)), encoding='utf-8')
    env = {k: v for k, v in os.environ.items()
           if k not in {'ACG_INTERPOLATE', 'ACG_TIMING_ONLY', 'ACG_BVH', 'ACG_SAMPLE_SEED',
                        'ACG_FELT_CACHE', 'ACG_FELT_LOCAL_SPP', 'ACG_LOCAL_SPP',
                        'ACG_BACKGROUND_SPP', 'ACG_RANDOM_LIGHT', 'ACG_LOCAL_SOBOL', 'ACG_RAIL_SOBOL'}}
    env['ACG_BVH'] = mode
    env['ACG_SAMPLE_SEED'] = str(args.reference_seed if folder.name == 'reference' else args.seed)
    if spacing:
        env['ACG_INTERPOLATE'] = str(spacing)
    with (folder / 'timings.csv').open('w') as timings, (folder / 'renderer.log').open('w') as log:
        subprocess.run([str(binary), '2', str(args.width), str(args.height), str(spp),
                        '--sequence', '0', str(manifest)], env=env, stdout=timings,
                       stderr=log, check=True)
    with (folder / 'timings.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(args.frames) or [int(r['frame']) for r in rows] != list(range(len(rows))):
        raise RuntimeError(f'Invalid frame timing sequence: {folder}')
    if {row['backend'] for row in rows} != {args.backend}:
        raise RuntimeError(f'Unexpected renderer backend: {folder}')
    return rows


def load_ppm(path):
    with path.open('rb') as f:
        if f.readline() != b'P6\n':
            raise ValueError(f'Expected binary PPM: {path}')
        w, h = map(int, f.readline().split())
        pixels = f.read() if f.readline() == b'255\n' else b''
        if w <= 0 or h <= 0 or len(pixels) != 3*w*h:
            raise ValueError(f'Invalid PPM: {path}')
        return (w, h), pixels


def ball_roi(args, frame):
    w, h = args.width, args.height
    scale = h / (2*max(.865, 1.54/(w/h)))  # Mirrors scene.h's camera_half_height.
    coords = list(map(float, (args.snapshots / f'frame_{frame:06d}.txt').read_text().split()))
    pixels = set()
    radius = .11*scale
    for x, z in zip(coords[::2], coords[1::2]):
        cx, cy = w/2 + scale*x, h/2 + scale*z
        for row in range(max(0, math.floor(cy-radius)), min(h, math.ceil(cy+radius))):
            for col in range(max(0, math.floor(cx-radius)), min(w, math.ceil(cx+radius))):
                if (col+.5-cx)**2 + (row+.5-cy)**2 <= radius**2:
                    pixels.add(row*w+col)
    return pixels


def quality(folder, ref, args):
    spatial = roi = temporal = 0
    roi_count = temporal_count = 0
    previous = previous_roi = None
    for j in range(args.warmup, len(args.frames)):
        size, image = load_ppm(folder / f'{j:04d}.ppm')
        ref_size, reference = load_ppm(ref / f'{j:04d}.ppm')
        if size != ref_size or size != (args.width, args.height):
            raise ValueError(f'Mismatched frame in {folder}: {j}')
        residual = [a-b for a, b in zip(image, reference)]
        region = ball_roi(args, args.frames[j])
        spatial += sum(d*d for d in residual)
        roi += sum(residual[3*i+c]**2 for i in region for c in range(3))
        roi_count += 3*len(region)
        if previous is not None and args.frames[j] == args.frames[j-1]+1:
            union = region | previous_roi
            temporal += sum((residual[3*i+c]-previous[3*i+c])**2 for i in union for c in range(3))
            temporal_count += 3*len(union)
        previous, previous_roi = residual, region
    def rmse(sumsq, count):
        return math.sqrt(sumsq/count)/255 if count else ''
    return (rmse(spatial, (len(args.frames)-args.warmup)*args.width*args.height*3),
            rmse(roi, roi_count), rmse(temporal, temporal_count))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'analytic', 'snapshots', 'output'):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--frames', type=int, nargs='+', default=list(range(91)))
    p.add_argument('--spp', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--spacing', type=int, nargs='+', default=[4],
                   help='interpolation grid spacings to compare; preserves variant names for the default')
    p.add_argument('--reference-spp', type=int, default=128)
    p.add_argument('--seed', type=int, default=0x63d83595)
    p.add_argument('--reference-seed', type=int, default=20261126)
    p.add_argument('--warmup', type=int, default=1)
    a = p.parse_args()
    if a.backend == 'cuda' and os.environ.get('ACG_GPU_ALLOCATION_APPROVED') != 'yes':
        p.error('CUDA requires an explicitly approved existing allocation')
    if not 0 <= a.warmup < len(a.frames) or min(a.frames) < 0 or min(a.spp) < 1 or a.reference_spp <= max(a.spp):
        p.error('Invalid frames, warmup, or sample budgets')
    if any(not 1 <= spacing <= 64 for spacing in a.spacing):
        p.error('interpolation spacing must be in [1, 64]')
    a.output.mkdir(parents=True)
    ref = a.output / 'reference'
    run(a.baseline, 'scan', 0, a.reference_spp, a, ref)
    variants = [('scan', a.baseline, 'scan', 0), ('rebuild', a.baseline, 'rebuild', 0),
                ('refit', a.baseline, 'refit', 0), ('interpolate', a.baseline, 'scan', 4),
                ('analytic', a.analytic, 'scan', 0), ('refit-interpolate', a.baseline, 'refit', 4),
                ('refit-analytic', a.analytic, 'refit', 0), ('interpolate-analytic', a.analytic, 'scan', 4),
                ('combined', a.analytic, 'refit', 4)]
    variants = [(name if a.spacing == [4] or not spacing else f'{name}-grid{grid}', binary, mode, grid)
                for name, binary, mode, spacing in variants
                for grid in (sorted(set(a.spacing)) if spacing else [0])]
    with (a.output / 'summary.csv').open('w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['variant', 'spp', 'frames', 'compute_p50_ms', 'compute_p95_ms',
                      'display_rgb_rmse', 'ball_neighborhood_rmse', 'adjacent_roi_residual_delta_rmse'])
        for spp in sorted(set(a.spp)):
            for name, binary, mode, spacing in variants:
                folder = a.output / f'{name}-{spp}'
                rows = run(binary, mode, spacing, spp, a, folder)
                times = sorted(float(row['total_wall_ms']) - float(row['ppm_wall_ms'])
                               for row in rows[a.warmup:])
                out.writerow([name, spp, len(times), statistics.median(times),
                              times[math.ceil(.95*len(times))-1], *quality(folder, ref, a)])
                f.flush()


if __name__ == '__main__':
    main()
