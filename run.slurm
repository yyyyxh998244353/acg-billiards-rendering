#!/bin/bash -l
#SBATCH --job-name=acg-stages
#SBATCH --partition=compute
#SBATCH --nodelist=i5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# Submit from the directory containing render.cu and scene.h. WIDTH, HEIGHT,
# SPP and SCENE_TIME can be changed together for all three comparisons.
set -euo pipefail
if [[ ${ACG_GPU_ALLOCATION_APPROVED:-} != yes ]]; then
    echo 'GPU execution requires explicit allocation approval; compile offline otherwise.' >&2
    exit 2
fi
src=${SLURM_SUBMIT_DIR:?requires Slurm}
job_dir="$src/runs/${SLURM_JOB_ID:?}"
mkdir -p "$src/runs"
mkdir "$job_dir"             # Never overwrite evidence from a previous run.
cd "$job_dir"
exec > >(tee job.log) 2>&1
width=${WIDTH:-960}; height=${HEIGHT:-540}; spp=${SPP:-32}; time=${SCENE_TIME:-0}
printf 'job_dir=%s\njob_id=%s\nnode=%s\nstart_utc=%s\ncuda_visible_devices=%s\n' \
    "$job_dir" "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'width=%s height=%s spp=%s scene_time=%s\n' "$width" "$height" "$spp" "$time"
printf '%s\n' 'exposure=1 tonemap=per-channel-Reinhard encoding=sRGB/P6 seed_base=0x63d83595'
sources=(render.cu scene.h felt_pyramid.h ball_bvh.h sequence.h interpolate.h soft_intervals.h physics.py)
for source in "${sources[@]}"; do cp -- "$src/$source" .; done
cp -- "$0" run.slurm
sha256sum "${sources[@]}" run.slurm > sources.sha256
python3 physics.py . | tee physics.log
sha256sum before.txt after.txt > snapshots.sha256
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv

# Prefer the chosen module or explicit compiler; many nodes install the
# toolkit under /usr/local without including nvcc on the batch PATH.
if [[ -n ${CUDA_MODULE:-} ]]; then module load "$CUDA_MODULE"; fi
nvcc=${NVCC:-nvcc}
if [[ -z ${NVCC:-} ]] && ! command -v nvcc >/dev/null; then
    for candidate in "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /opt/cuda/bin/nvcc; do
        if [[ -x $candidate ]]; then nvcc=$candidate; break; fi
    done
fi
if ! command -v "$nvcc" >/dev/null; then
    echo 'nvcc not found on the allocated node; select a toolkit with CUDA_MODULE or NVCC.' >&2
    exit 1
fi
command -v "$nvcc"
"$nvcc" --version
build=("$nvcc" -O3 -std=c++17 "-arch=${CUDA_ARCH:-sm_90}" render.cu -o render)
printf 'build:'; printf ' %q' "${build[@]}"; printf '\n'
"${build[@]}"

names=(point materials area)
for stage in 0 1 2; do
    for snapshot in before after; do
        stem="stage${stage}_${names[$stage]}_${snapshot}"
        cmd=(./render "$stage" "$width" "$height" "$spp" "$stem.ppm" "$time" "$snapshot.txt")
        printf 'render:'; printf ' %q' "${cmd[@]}"; printf '\n'
        "${cmd[@]}" > "$stem.csv"
        cat "$stem.csv"
        [[ -f timings.csv ]] || printf 'snapshot,%s\n' "$(head -n 1 "$stem.csv")" > timings.csv
        printf '%s,%s\n' "$snapshot" "$(tail -n 1 "$stem.csv")" >> timings.csv
        sha256sum "$stem.ppm" >> images.sha256
    done
done
printf 'end_utc=%s\n' "$(date -u +%FT%TZ)"
