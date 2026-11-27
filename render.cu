// CPU:  g++ -O3 -std=c++17 -x c++ render.cu -o render_cpu
// H100: nvcc -O3 -std=c++17 -arch=sm_90 render.cu -o render
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <new>
#define RENDER_HD __host__ __device__
#else
#define RENDER_HD
#endif
#include "scene.h"
using acg_scene_detail::BallBVH;
using acg_scene_detail::Balls;
#include "sequence.h"
#include "interpolate.h"

namespace {
using Clock = std::chrono::steady_clock;
#ifdef ACG_ANALYTIC_SHADOW
constexpr bool analytic_shadow=true;
#else
constexpr bool analytic_shadow=false;
#endif
using u32 = std::uint32_t;
constexpr u32 base_seed = 0x63d83595u;
struct RGB8 { unsigned char r, g, b; };
static_assert(sizeof(RGB8) == 3, "P6 needs packed RGB triplets");

RENDER_HD inline u32 mix_bits(u32 v) {
    v ^= v >> 16; v *= 0x7feb352du;
    v ^= v >> 15; v *= 0x846ca68bu;
    return v ^ (v >> 16);
}

struct RowSeed {
    u32 sample_key, path_key;
};

RENDER_HD inline RowSeed row_seed(int y,u32 sample_seed) {
    const auto row = static_cast<u32>(y);
    return {mix_bits(sample_seed ^ row), acg_scene_detail::hash32(row + 0x243f6a88u)};
}

RENDER_HD inline float display_code(float t) {
    const float s = t <= 0.0031308f ? 12.92f * t : 1.055f * powf(t, 1.0f / 2.4f) - 0.055f;
    return 255.0f * s + 0.5f;
}

RENDER_HD inline unsigned char display_reference(float linear) {
    if (!(linear > 0.0f)) return 0;
    // Unit exposure, per-channel Reinhard, then the piecewise sRGB transfer.
    const float t = linear > 1.0f ? 1.0f - 1.0f / (1.0f + linear)
                                  : linear / (1.0f + linear);
    return static_cast<unsigned char>(display_code(t));
}

#if defined(__APPLE__) && defined(ACG_PACKED_ROOM)
constexpr u32 first_color_bits = 111u << 23; // 2^-16, below the first code boundary.
constexpr u32 last_color_bits = 139u << 23; // 2^12, past the last code boundary.
constexpr int color_shift = 14; // Keep the exponent and the top nine fraction bits.
constexpr int color_bins = (last_color_bits - first_color_bits) >> color_shift;
constexpr std::uint16_t uncertain_color = 256;

inline float color_from_bits(u32 bits) {
    float value;
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

const auto linear_colors = [] {
    std::array<std::uint16_t,color_bins> table;
    for (int i = 0; i < color_bins; ++i) {
        const u32 start = first_color_bits + (static_cast<u32>(i) << color_shift);
        const float lo = color_from_bits(start);
        const float hi = color_from_bits(start + (1u << color_shift) - 1);
        // Bound the quotient with independent numerator and denominator bounds.
        // The upper Reinhard branch is monotone under rounded float arithmetic.
        const float low_code = display_code(lo < 1.f ? lo / (1.f + hi) : 1.f - 1.f / (1.f + lo));
        const float high_code = display_code(lo < 1.f ? hi / (1.f + lo) : 1.f - 1.f / (1.f + hi));
        const auto code = static_cast<std::uint16_t>(low_code);
        // Leave room for libm rounding and the slight discontinuity at the sRGB knee.
        constexpr float margin = 0x1p-10f; // In byte units.
        table[i] = low_code > code + margin && high_code < code + 1 - margin ? code : uncertain_color;
    }
    return table;
}();
#endif

RENDER_HD inline unsigned char display_byte(float linear) {
#if defined(__APPLE__) && defined(ACG_PACKED_ROOM)
    u32 bits;
    std::memcpy(&bits,&linear,sizeof(bits));
    // The unsigned range check also sends negative and nonfinite values to the formula.
    const u32 index = (bits - first_color_bits) >> color_shift;
    if (index < static_cast<u32>(color_bins)) {
        const auto code = linear_colors[index];
        if (code != uncertain_color) return static_cast<unsigned char>(code);
    }
#endif
    return display_reference(linear);
}

RENDER_HD inline RGB8 render_pixel(int x,int y,int w,int h,int stage,int spp,const Balls* balls,unsigned poses,RowSeed row,float cached) {
    // No dependency on launch order, stage, spp budget or time. Matching pixels
    // share the same prefix of random samples in all comparison renders.
    using namespace acg_scene_detail;
    const auto column = static_cast<u32>(x);
    const u32 first = mix_bits(row.sample_key ^ column);
    const u32 path_key = row.path_key ^ hash32(column + 0x9e3779b9u);
    float r = 0, g = 0, b = 0;
    for (int s = 0; s < spp; ++s) {
        const u32 seed = mix_bits(first + static_cast<u32>(s));
        const unsigned pose = (static_cast<std::uint64_t>(mix_bits(seed ^ 0xba7c19u)) * poses) >> 32;
        Rng rng={hash32((seed?seed:base_seed)^path_key)};
        const Ray ray=camera(x,y,w,h,stage,rng);
        const Vec v=trace(ray,balls[pose],stage,0.f,rng,analytic_shadow,cached);
        r += v.x; g += v.y; b += v.z;
    }
    const float inv = 1.0f / spp;
    return {display_byte(r * inv), display_byte(g * inv), display_byte(b * inv)};
}

#ifdef __CUDACC__
template<bool SharedPose>
__global__ void render_pixels(RGB8* pixels,size_t count,int w,int h,int stage,int spp,const Balls* device_balls,unsigned poses,u32 sample_seed,const float* cache) {
    if constexpr (SharedPose) {
        __shared__ BallBVH block_tree;
        alignas(Balls) __shared__ unsigned char pose_storage[sizeof(Balls)];
        __shared__ const Balls* block_pose;
        const unsigned node = threadIdx.x;
        if (node < BallBVH::node_count) {
            block_tree.box[node] = device_balls->tree->box[node];
            block_tree.tag[node] = device_balls->tree->tag[node];
        }
        if (node == 0) {
            auto* pose = ::new (pose_storage) Balls(*device_balls);
            pose->tree = &block_tree;
            block_pose = pose;
        }
        __syncthreads();
        device_balls = block_pose;
    }
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        const int x = static_cast<int>(i % w), y = static_cast<int>(i / w);
        pixels[i] = render_pixel(x,y,w,h,stage,spp,device_balls,poses,row_seed(y,sample_seed),cache?cache[i]:-1.f);
    }
}

void check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(call) check((call), #call)
#endif

int integer(const char* text, int low, int high, const char* name) {
    int v;
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, v);
    if (result.ec != std::errc{} || result.ptr != end || v < low || v > high)
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    return v;
}

float scene_time(const char* text) {
    char* end = nullptr;
    errno = 0;
    const float v = std::strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(v))
        throw std::invalid_argument(std::string("invalid time: ") + text);
    return v;
}

struct Receiver : acg_interpolate::Guide { Vec point; };

constexpr Vec guide_direction={0,-1,0};
constexpr float contact_distance=.13f;
static_assert(2*acg_scene_detail::radius*acg_scene_detail::radius < contact_distance*contact_distance);

// The guide point lies on y=0. place_balls() fixes every center at y=radius.
inline Receiver probe_receiver(Vec point,const Balls& balls,const acg_scene_detail::Box& contact_bounds,
                               Vec inverse,int stage) {
    using namespace acg_scene_detail;
    Receiver guide{};
    guide.point=point;
    // The contact check also excludes every possible ball hit on this ray.
    if(point.x>=contact_bounds.a.x && point.x<=contact_bounds.b.x &&
       point.z>=contact_bounds.a.z && point.z<=contact_bounds.b.z) {
        for(const Vec center:balls.c) {
            const Vec offset=point-center;
            if(dot(offset,offset)<contact_distance*contact_distance) return guide;
        }
    }
    float depth=camera_height;
    if(!(overhead_slate_clear && fabsf(point.x)<table_clear_x && fabsf(point.z)<table_clear_z)) {
        const Ray ray={point+Vec{0,camera_height,0},guide_direction};
        // Farther hits cannot pass the receiver height check.
        constexpr float limit=camera_height+2*eps;
#if !defined(__CUDA_ARCH__)
        const bool interior=inside_room(ray.o,inverse,limit);
#else
        constexpr bool interior=false;
#endif
        Hit hit;
        if(!intersect_with_inverse(ray,inverse,balls,stage,eps,limit,hit,interior) ||
           hit.mat!=Felt || hit.normal.y!=1 || fabsf(hit.t-camera_height)>eps) return guide;
        depth=hit.t;
    }
    return {{true,0,Felt,depth,{0,1,0}},point};
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace

int main(int argc, char** argv) try {
    if (argc < 8 || argc > 71) {
        std::cerr << "Usage: " << argv[0] << " stage(0..2) width height spp output.ppm time snapshot.txt [shutter_snapshot.txt ...]\n";
        return 2;
    }
    const int imax = std::numeric_limits<int>::max();
    const int stage = integer(argv[1], 0, 2, "stage");
    const int w = integer(argv[2], 1, imax, "width"), h = integer(argv[3], 1, imax, "height");
    const int spp = integer(argv[4], 1, imax, "spp");
    const float time = scene_time(argv[6]);
    const char* seed_env=std::getenv("ACG_SAMPLE_SEED");
    const u32 sample_seed=seed_env?static_cast<u32>(integer(seed_env,0,imax,"sample seed")):base_seed;
    const bool sequence_mode=argc==8 && std::strcmp(argv[5],"--sequence")==0;
    const char* timing=std::getenv("ACG_TIMING_ONLY");
    const bool timing_only=timing && std::strcmp(timing,"1")==0;
    const auto frames=sequence_mode?sequence::read_manifest(argv[7]):std::vector<sequence::Frame>{};
    float balls[64 * 32];
    const unsigned poses=sequence_mode?1:argc-7;
    BallBVH prior{};
    const std::uint64_t count64 = static_cast<std::uint64_t>(w) * h;
    if (count64 > std::numeric_limits<size_t>::max() / sizeof(RGB8) ||
        count64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(RGB8))
        throw std::length_error("image byte count exceeds platform limit");
    const size_t count = static_cast<size_t>(count64), bytes = count * sizeof(RGB8);
    std::unique_ptr<RGB8[]> image(new RGB8[count]);
    std::vector<float> cache;
#ifdef __CUDACC__
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "CUDA device 0: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    BallBVH* device_trees=nullptr;
    float* device_cache=nullptr;
    Balls* device_balls=nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_balls),poses*sizeof(Balls)));
    RGB8* device = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    constexpr unsigned threads = 128;
    static_assert(threads >= BallBVH::node_count);
#endif
    for(size_t frame=0;frame<(sequence_mode?frames.size():1);++frame) {
    for(unsigned i=0;i<poses;++i) {
        const auto data=sequence::load_snapshot(sequence_mode?frames[frame].snapshot_path:std::filesystem::path(argv[7+i]));
        std::copy(data.begin(),data.end(),balls+32*i);
    }
    const auto total_start = Clock::now();
    BallBVH trees[64];
    const char* query=std::getenv("ACG_BVH");
    const bool use_tree=query && (std::strcmp(query,"rebuild")==0 || std::strcmp(query,"refit")==0);
    if(query && !use_tree && std::strcmp(query,"scan")!=0)
        throw std::runtime_error("ACG_BVH must be scan, rebuild or refit");
    Balls prepared[64];
#ifdef ACG_PACKED_ROOM
    acg_scene_detail::PackedBalls packed_balls[64];
#endif
    for(unsigned i=0;i<poses;++i) {
        auto& state=prepared[i];
        place_balls(state,time,balls+32*i);
        if(!use_tree) {
#ifdef ACG_PACKED_ROOM
            acg_scene_detail::prepare_packed_balls(state,packed_balls[i]);
#endif
            continue;
        }
        if((i || frame) && std::strcmp(query,"refit")==0) {
            trees[i]=i?trees[i-1]:prior;
            refit_ball_bvh(trees[i],state);
        } else build_ball_bvh(trees[i],state);
        prior=trees[i];
        state.tree=trees+i;
    }
    const double prepare_ms=elapsed_ms(total_start);
    std::cerr << "balls=" << (query?query:"scan") << " prepare_ms=" << prepare_ms << '\n';
    const auto cache_start=Clock::now();
    if(const char* setting=std::getenv("ACG_INTERPOLATE"); setting && poses==1) {
        using namespace acg_scene_detail;
        using namespace acg_interpolate;
        const Balls& state=prepared[0];
        Settings config;
        config.spacing=integer(setting,1,64,"interpolation grid spacing");
        cache.assign(count,-1.f);
        const float aspect=float(w)/h, half_height=hi(.77f,1.40f/aspect);
        const Vec guide_inverse=invdir(guide_direction);
        const Box contact_bounds={state.box.a-grey(contact_distance),state.box.b+grey(contact_distance)};
        auto guide=[&](int x,int y) noexcept {
            const Vec point={(2*(x+.5f)/w-1)*aspect*half_height,0,(2*(y+.5f)/h-1)*half_height};
            return probe_receiver(point,state,contact_bounds,guide_inverse,stage);
        };
        auto exact=[&](int,int,const Receiver& g) noexcept {
            if(!g.smooth_diffuse) return 0.f;
            return visibility_to_light(g.point+Vec{0,eps,0},point_light_position,state,stage);
        };
        const auto stat=fixed_grid<false,true>(w,h,config,guide,exact,[&](int x,int y,const Receiver&,float value,Source) noexcept {
            cache[size_t(y)*w+x]=value;
        });
        std::cerr << "interpolated=" << stat.interpolated << " pixels=" << count
                  << " cache_ms=" << elapsed_ms(cache_start) << '\n';
    }
    const double cache_ms=elapsed_ms(cache_start);
#ifdef __CUDACC__
    const auto upload_start=Clock::now();
    if(use_tree) {
        if(!device_trees) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_trees),poses*sizeof(BallBVH)));
        CUDA_CHECK(cudaMemcpy(device_trees,trees,poses*sizeof(BallBVH),cudaMemcpyHostToDevice));
    }
    for(unsigned i=0;i<poses;++i) prepared[i].tree=use_tree?device_trees+i:nullptr;
    CUDA_CHECK(cudaMemcpy(device_balls,prepared,poses*sizeof(Balls),cudaMemcpyHostToDevice));
    if(!cache.empty()) {
        if(!device_cache) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cache),count*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(device_cache,cache.data(),count*sizeof(float),cudaMemcpyHostToDevice));
    }
    const double upload_ms=elapsed_ms(upload_start);
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count - 1) / threads + 1, 65535));
    CUDA_CHECK(cudaEventRecord(begin));
    if (poses == 1 && use_tree)
        render_pixels<true><<<blocks, threads>>>(device,count,w,h,stage,spp,device_balls,poses,sample_seed,device_cache);
    else
        render_pixels<false><<<blocks, threads>>>(device,count,w,h,stage,spp,device_balls,poses,sample_seed,device_cache);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float render_ms;
    CUDA_CHECK(cudaEventElapsedTime(&render_ms, begin, end));
    const auto copy_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(image.get(), device, bytes, cudaMemcpyDeviceToHost));
    const double d2h_ms = elapsed_ms(copy_start);
    constexpr const char* backend = "cuda";
#else
    constexpr double upload_ms=0;
    const auto render_start = Clock::now();
#ifdef ACG_COUNT_TRAVERSAL
    acg_scene_detail::ball_counters={};
#endif
    RGB8* const pixels = image.get();
    const float* const shadow_cache = cache.empty()?nullptr:cache.data();
#ifdef _OPENMP
    // Shrink chunks near the end to balance expensive rows across workers.
    #pragma omp parallel for schedule(guided, 1)
#endif
    for (int y = 0; y < h; ++y) {
        const size_t offset = static_cast<size_t>(y) * w;
        RGB8* const row = pixels + offset;
        const float* const shadows = shadow_cache?shadow_cache + offset:nullptr;
        const RowSeed seed = row_seed(y,sample_seed);
        for (int x = 0; x < w; ++x)
            row[x] = render_pixel(x,y,w,h,stage,spp,prepared,poses,seed,shadows?shadows[x]:-1.f);
    }
    const double render_ms = elapsed_ms(render_start), d2h_ms = 0.0;
#ifdef ACG_COUNT_TRAVERSAL
    const auto& counters=acg_scene_detail::ball_counters;
    std::cerr << "traversal," << frame << ',' << counters.rays << ','
              << counters.boxes << ',' << counters.spheres << '\n';
#endif
    constexpr const char* backend = "cpu";
#endif
    double io_ms=0;
    if(!timing_only) {
        const auto io_start = Clock::now();
        std::ofstream out(sequence_mode?frames[frame].output_path:std::filesystem::path(argv[5]),std::ios::binary);
        out << "P6\n" << w << ' ' << h << "\n255\n";
        out.write(reinterpret_cast<const char*>(image.get()), static_cast<std::streamsize>(bytes));
        out.close();
        if (!out) throw std::runtime_error("failed writing output image");
        io_ms=elapsed_ms(io_start);
    }
    const double total_ms=elapsed_ms(total_start);
    // render_ms = one kernel (sampling + tonemap) on CUDA, render loop on CPU.
    // Frame wall time excludes manifest/snapshot parsing and persistent device setup.
    if(frame==0) std::cout << "backend,stage,width,height,spp,time,base_seed,render_ms,d2h_wall_ms,ppm_wall_ms,total_wall_ms,frame,prepare_ms,cache_ms,upload_ms\n";
    std::cout
              << backend << ',' << stage << ',' << w << ',' << h << ',' << spp << ',' << std::setprecision(9)
              << time << ',' << sample_seed << ',' << std::fixed << std::setprecision(3) << render_ms << ','
              << d2h_ms << ',' << io_ms << ',' << total_ms << ',' << frame << ',' << prepare_ms << ',' << cache_ms << ',' << upload_ms << '\n';
    }
#ifdef __CUDACC__
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFree(device_balls));
    if(device_trees) CUDA_CHECK(cudaFree(device_trees));
    if(device_cache) CUDA_CHECK(cudaFree(device_cache));
#endif
    return 0;
} catch (const std::exception& e) {
    std::cerr << "render: " << e.what() << '\n';
    return 1;
}
