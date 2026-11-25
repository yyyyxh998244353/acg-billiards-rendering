"""Generate physics snapshots, render a replay, and encode it without Slurm submission."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('renderer', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--spp', type=int, default=2)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--last-frame', type=int, default=90)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', choices=('all','prepare','render','encode'), default='all')
    args = parser.parse_args()
    if min(args.width, args.height, args.spp, args.fps, args.last_frame) < 1 or args.fps > 120:
        parser.error('positive dimensions, spp, and last frame required; fps must be 1–120')
    renderer, out = args.renderer.resolve(), args.output.resolve()
    manifest = out/'frames.txt'
    if args.phase in ('prepare','all'):
        out.mkdir(parents=True)
        source = Path(__file__).resolve().with_name('physics.py')
        with (out/'physics.json').open('w') as metadata:
            subprocess.run([sys.executable, str(source), str(out/'snapshots'), '--seed', str(args.seed),
                            '--fps', str(args.fps), '--frames', *map(str, range(args.last_frame+1))],
                           stdout=metadata, check=True)
        manifest.write_text(''.join(f'snapshots/frame_{i:06d}.txt frame-{i:04d}.ppm\n'
                                   for i in range(args.last_frame+1)), encoding='ascii')
    if args.phase == 'prepare':
        return
    if args.phase in ('render','all'):
        env = dict(os.environ)
        env.pop('ACG_TIMING_ONLY', None)
        with (out/'timings.csv').open('w') as timings:
            subprocess.run([str(renderer), '2', str(args.width), str(args.height), str(args.spp),
                            '--sequence', '0', str(manifest)], env=env, stdout=timings, check=True)
    with (out/'timings.csv').open() as timings:
        rows = list(csv.DictReader(timings))
    if [int(row['frame']) for row in rows] != list(range(args.last_frame+1)):
        raise RuntimeError('renderer did not report the expected sequence')
    if args.phase == 'render':
        return
    from PIL import Image
    frames = []
    for i in range(args.last_frame+1):
        with Image.open(out/f'frame-{i:04d}.ppm') as image:
            if image.size != (args.width, args.height):
                raise RuntimeError(f'Unexpected dimensions at frame {i}')
            frames.append(image.convert('RGB'))
    durations = [round((i+1)*1000/args.fps)-round(i*1000/args.fps) for i in range(len(frames))]
    frames[0].save(out/'replay.webp', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, quality=85)
    print(f'{len(frames)} frames: {out / "replay.webp"}')


if __name__ == '__main__':
    main()
