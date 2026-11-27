"""Overlay ball-center paths on ortho-cpu.png rendered with this revision's camera."""
from pathlib import Path
from PIL import Image, ImageDraw

out = Path('outputs')
frames = [list(zip(*(iter(map(float, p.read_text().split())),) * 2))
          for p in sorted((out / 'trajectory').glob('frame_*.txt'))]
assert len(frames) == 91 and all(len(frame) == 16 for frame in frames)
with Image.open(out / 'ortho-cpu.png') as pair:
    w, total = pair.size
    h = total // 2
    before = pair.crop((0, 0, w, h))
    after = pair.crop((0, h, w, total))
    before.save(out / 'overhead-before.png')
    after.save(out / 'overhead-after.png')
scale = h / (2 * max(.865, 1.54 / (w / h)))  # Mirrors scene.h's camera_half_height.
canvas = after.resize((w * 2, h * 2)).convert('RGBA')
ink = Image.new('RGBA', canvas.size)
draw = ImageDraw.Draw(ink)
colors = ['white', '#ffcc52', '#91caff', '#ff9bad', '#c4b5fd', '#a7efa0', '#ffc085', '#72e4ed']
for i, path in enumerate(zip(*frames)):
    xy = [(round(w + 2*x*scale), round(h + 2*z*scale)) for x, z in path]
    draw.line(xy, fill=colors[i % len(colors)], width=3, joint='curve')
result = Image.alpha_composite(canvas, ink).convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
result.save(out / 'overhead-trajectory.png')
