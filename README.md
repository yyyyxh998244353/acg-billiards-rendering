# 台球开球 · CUDA 路径追踪

这是一个高等计算机图形学课程项目：用 Python 模拟 16 颗台球从开球到散开的轨迹，再用自己实现的 C++ / CUDA 路径追踪器逐帧生成俯视画面。球、木轨倒角和球袋内衬都用解析几何表示；台布、木纹和球面号码用程序生成，不需要下载模型或贴图。项目做的是**离线渲染和渲染性能实验**，不是交互式台球游戏。

## 不用显卡，先渲染两张图

需要 Python 3 和支持 C++17 的编译器（Linux/macOS）。从仓库根目录运行：

```sh
c++ -O3 -std=c++17 -x c++ render.cu -o render_cpu
python3 physics.py outputs/first-look --seed 42 --fps 60 --frames 0 90
./render_cpu 2 160 90 8 outputs/first-look/before.ppm 0 outputs/first-look/before.txt
./render_cpu 2 160 90 8 outputs/first-look/after.ppm 0 outputs/first-look/after.txt
```

`before.ppm` 是开球前，`after.ppm` 是 1.5 秒后。命令里的 `2` 选择带材质、点光源和矩形面光源的完整渲染路径；`8` 是每像素的采样数（spp）。提高 spp 可以减少颗粒，提高分辨率可以看清更多细节；两者都会增加渲染时间。

想看球滚动，可以安装 [Pillow](https://pillow.readthedocs.io/en/stable/installation/basic-installation.html) 后，生成一小段 WebP 动画，在浏览器里打开 `outputs/quick-replay/replay.webp`。每次使用一个新的输出目录。

```sh
python3 preview_replay.py ./render_cpu outputs/quick-replay \
  --width 128 --height 72 --fps 30 --last-frame 45 --spp 2
```

## 用 CUDA 重现 1080p 成片

成片用单张 NVIDIA H100 离线渲染：**1920×1080，48 spp，30 FPS，6 秒**。下面的命令生成 180 张 PPM，再用 FFmpeg 编码为 MP4。`sm_90` 是 H100 的目标架构；其他 GPU 要改为对应的架构。

```sh
nvcc -O3 -std=c++17 -arch=sm_90 --maxrregcount=128 render.cu -o render_gpu
python3 preview_replay.py ./render_gpu outputs/break-1080p --phase prepare \
  --width 1920 --height 1080 --fps 30 --last-frame 179 --spp 48
ACG_BVH=rebuild ACG_SAMPLE_SEED=6035729 \
  python3 preview_replay.py ./render_gpu outputs/break-1080p --phase render \
  --width 1920 --height 1080 --fps 30 --last-frame 179 --spp 48
ffmpeg -framerate 30 -i outputs/break-1080p/frame-%04d.ppm \
  -c:v libx264 -pix_fmt yuv420p outputs/break-1080p.mp4
```

集群上只需在 GPU 作业中运行 `--phase render`；编译、轨迹生成和编码不需要 GPU。脚本不会自行提交 Slurm 作业。`timings.csv` 分别记录场景准备、上传、渲染、回读和写盘时间。

如果想重现**分层降噪版**，用一个新输出目录，把上面的 `--spp 48` 改为 `--spp 96`，并在 `--phase render` 的命令前加 `ACG_FELT_CACHE=384 ACG_FELT_LOCAL_SPP=24`。它用高采样锚点重建台布的低频漫反射光照；球、球袋和半影仍走完整路径。该模式只支持 CUDA 单姿态 stage 2，不与其他阴影缓存叠加。

在新版灯光下复现**球周优先版**，改用 `--spp 72`，渲染阶段设置 `ACG_FELT_CACHE=320 ACG_FELT_LOCAL_SPP=20 ACG_LOCAL_SPP=144`。额外样本只分配给球周，默认首交面灯使用低差异序列。

新版轨盖和袋衬参考实物装配；轨口内的包布、石板台阶和袋衬各有真实求交面。复现新版成片时可在上一条配置中加 `ACG_BACKGROUND_SPP=24`，只降低带安全边距的桌外像素预算。

## 代码从哪里读

- [`physics.py`](physics.py)：可重复的开球模拟，输出各帧的球心坐标。
- [`scene.h`](scene.h)：球桌、球袋、球体、程序材质、相机和路径积分。渲染器根据球的位置估算纹理转动，并计算镜面反射和阴影。
- [`render.cu`](render.cu) 和 [`ball_bvh.h`](ball_bvh.h)：CPU/CUDA 渲染入口，球体相交加速，以及输出计时。可用 `ACG_BVH=scan|rebuild|refit` 对比线扫、逐帧建树和仅更新包围盒。
- [`check_soft_intervals.cpp`](check_soft_intervals.cpp)：几何和光照回归，覆盖袋口、木轨接缝、遮挡和软阴影积分。

## 实测结果

- 上述 **1080p、48 spp 成片**在 H100 上的单帧中位耗时约 **1.19 秒**（含 PPM 写盘，不含编码）。这是离线渲染，不是 30 FPS 实时运行。
- 两帧同场景对照中，48 spp 相比 24 spp 的全图 sRGB RMSE 约低 **21%**，参考为独立随机种子的 128 spp 图像；试过的轻量双边滤波使全图误差略升，未用于成片。
- 在较早的 H100 高采样对照中，逐帧重建球体 BVH 比线扫的配对整帧耗时低约 **9–12%**（1080p、8 spp），两种方法的对照帧逐字节一致。这个对照与成片使用的是不同版本。

详见 [H100 实验与成片记录](.agents/docs/h100-hybrid-sampling.md)、[早期基线](EXPERIMENTS.md) 和 [项目设计计划](PLAN.md)。实验记录中的 `outputs/` 是本地生成目录，不随仓库分发。

本仓库公开供查看，未附带开源许可证。
