"""Run reproducible renderer ablations; never submits Slurm jobs."""
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
           if k not in {'ACG_INTERPOLATE', 'ACG_TIMING_ONLY', 'ACG_BVH'}}
    env['ACG_BVH'] = mode
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


def error(image_path, reference_path):
    def load(path):
        with path.open('rb') as f:
            if f.readline() != b'P6\n':
                raise ValueError(f'Expected binary PPM: {path}')
            w, h = map(int, f.readline().split())
            pixels = f.read() if f.readline() == b'255\n' else b''
            if w <= 0 or h <= 0 or len(pixels) != 3*w*h:
                raise ValueError(f'Invalid PPM: {path}')
            return (w, h), pixels
    size, image = load(image_path)
    ref_size, ref = load(reference_path)
    if size != ref_size:
        raise ValueError(f'Mismatched reference: {image_path}')
    return sum((a-b)**2 for a, b in zip(image, ref)) / (len(ref)*255**2)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'analytic', 'snapshots', 'output'):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--frames', type=int, nargs='+', default=list(range(91)))
    p.add_argument('--spp', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--reference-spp', type=int, default=128)
    p.add_argument('--warmup', type=int, default=1)
    a = p.parse_args()
    if a.backend == 'cuda' and os.environ.get('ACG_GPU_ALLOCATION_APPROVED') != 'yes':
        p.error('CUDA requires an explicitly approved existing allocation')
    if not 0 <= a.warmup < len(a.frames) or min(a.frames) < 0 or min(a.spp) < 1 or a.reference_spp <= max(a.spp):
        p.error('Invalid frames, warmup, or sample budgets')
    a.output.mkdir(parents=True)
    ref = a.output / 'reference'
    run(a.baseline, 'scan', 0, a.reference_spp, a, ref)
    variants = [('scan', a.baseline, 'scan', 0), ('rebuild', a.baseline, 'rebuild', 0),
                ('refit', a.baseline, 'refit', 0), ('interpolate', a.baseline, 'scan', 4),
                ('analytic', a.analytic, 'scan', 0), ('refit-interpolate', a.baseline, 'refit', 4),
                ('refit-analytic', a.analytic, 'refit', 0), ('interpolate-analytic', a.analytic, 'scan', 4),
                ('combined', a.analytic, 'refit', 4)]
    with (a.output / 'summary.csv').open('w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['variant', 'spp', 'frames', 'compute_p50_ms', 'compute_p95_ms', 'display_rgb_rmse'])
        for spp in sorted(set(a.spp)):
            for name, binary, mode, spacing in variants:
                folder = a.output / f'{name}-{spp}'
                rows = run(binary, mode, spacing, spp, a, folder)
                times = sorted(float(row['total_wall_ms']) - float(row['ppm_wall_ms'])
                               for row in rows[a.warmup:])
                mse = [error(folder / f'{j:04d}.ppm', ref / f'{j:04d}.ppm')
                       for j in range(a.warmup, len(rows))]
                out.writerow([name, spp, len(times), statistics.median(times),
                              times[math.ceil(.95*len(times))-1], math.sqrt(statistics.mean(mse))])
                f.flush()


if __name__ == '__main__':
    main()
