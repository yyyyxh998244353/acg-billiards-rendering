from PIL import Image, ImageChops, ImageStat
from pathlib import Path
import math, csv
p = Path('outputs/sweep')
ref = Image.open(p/'spp256.ppm').convert('RGB')
out = Image.new('RGB',(ref.width*2,ref.height*2))
print('spp,camera_paths,render_ms,display_rgb_rmse_vs_256')
for i,n in enumerate((4,16,64,256)):
    im=Image.open(p/f'spp{n}.ppm').convert('RGB')
    rmse=math.sqrt(sum(v*v for v in ImageStat.Stat(ImageChops.difference(im,ref)).rms)/3)/255
    with (p/f'spp{n}.csv').open() as f: row=next(csv.DictReader(f))
    print(f'{n},{n*im.width*im.height},{row["render_ms"]},{rmse:.6f}')
    out.paste(im,((i%2)*im.width,(i//2)*im.height))
out.save('outputs/sample-sweep.png')
