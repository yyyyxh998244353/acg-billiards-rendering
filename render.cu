// CPU:  g++ -O3 -std=c++17 -x c++ render.cu -o render_cpu
// H100: nvcc -O3 -std=c++17 -arch=sm_90 render.cu -o render
#include <algorithm>
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

RENDER_HD inline unsigned char display_byte(float linear) {
    if (!(linear > 0.0f)) return 0;
    // Unit exposure, per-channel Reinhard, then the piecewise sRGB transfer.
    const float t = linear > 1.0f ? 1.0f - 1.0f / (1.0f + linear)
                                  : linear / (1.0f + linear);
    const float s = t <= 0.0031308f ? 12.92f * t : 1.055f * powf(t, 1.0f / 2.4f) - 0.055f;
    return static_cast<unsigned char>(255.0f * s + 0.5f);
}

RENDER_HD inline RGB8 render_pixel(int x, int y, int w, int h, int stage, int spp, const Balls* balls, unsigned poses,float cached=-1) {
    // No dependency on launch order, stage, spp budget or time. Matching pixels
    // share the same prefix of random samples in all comparison renders.
    const u32 pixel_key = mix_bits(base_seed ^ static_cast<u32>(y)) ^ static_cast<u32>(x);
    const u32 first = mix_bits(pixel_key);
    float r = 0, g = 0, b = 0;
    for (int s = 0; s < spp; ++s) {
        const u32 seed = mix_bits(first + static_cast<u32>(s));
        const unsigned pose = (static_cast<std::uint64_t>(mix_bits(seed ^ 0xba7c19u)) * poses) >> 32;
        using namespace acg_scene_detail;
        Rng rng={hash32((seed?seed:base_seed)^hash32(u32(x)+0x9e3779b9u)^hash32(u32(y)+0x243f6a88u))};
        const Ray ray=camera(x,y,w,h,stage,rng);
        const Vec v=trace(ray,balls[pose],stage,0.f,rng,analytic_shadow,cached);
        r += v.x; g += v.y; b += v.z;
    }
    const float inv = 1.0f / spp;
    return {display_byte(r * inv), display_byte(g * inv), display_byte(b * inv)};
}

#ifdef __CUDACC__
__global__ void render_pixels(RGB8* pixels, size_t count, int w, int h, int stage, int spp, const Balls* device_balls,unsigned poses,const float* cache) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
        pixels[i] = render_pixel(static_cast<int>(i % w), static_cast<int>(i / w), w, h, stage, spp, device_balls, poses,cache?cache[i]:-1.f);
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
    for(unsigned i=0;i<poses;++i) {
        auto& state=prepared[i];
        place_balls(state,time,balls+32*i);
        if(!use_tree) continue;
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
        Balls state;
        place_balls(state,time,balls);
        state.tree=use_tree?trees:nullptr;
        Settings config;
        config.spacing=integer(setting,1,64,"interpolation grid spacing");
        cache.assign(count,-1.f);
        struct Receiver : Guide { Vec point; };
        auto guide=[&](int x,int y) {
            Receiver g{};
            float a=float(w)/h, v=hi(.77f,1.40f/a);
            g.point={(2*(x+.5f)/w-1)*a*v,0,(2*(y+.5f)/h-1)*v};
            constexpr float height=light_y-.12f;
            Hit hit;
            if(!intersect({g.point+Vec{0,height,0},{0,-1,0}},state,stage,eps,far_t,hit) ||
               hit.mat!=Felt || hit.normal.y!=1 || fabsf(hit.t-height)>eps) return g;
            // Exclude contacts and their nearby projected point-light shadows.
            for(const auto& c:state.c)
                if(dot(g.point-c,g.point-c)<.13f*.13f) return g;
            g.smooth_diffuse=true; g.object=0; g.material=Felt; g.depth=hit.t; g.normal={0,1,0};
            return g;
        };
        auto exact=[&](int,int,const Receiver& g) -> RGB {
            if(!g.smooth_diffuse) return {};
            Vec p=g.point+Vec{0,eps,0}, d=Vec{-.73f,1.54f,-.46f}-p;
            float l=sqrtf(dot(d,d)), value=visibility({p,d*(1/l)},l-3*eps,state,stage);
            return {value,value,value};
        };
        const auto stat=fixed_grid(w,h,config,guide,exact,[&](int x,int y,const Receiver&,RGB value,Source source) {
            if(source==Source::interpolated) cache[size_t(y)*w+x]=value[0];
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
    render_pixels<<<blocks, threads>>>(device, count, w, h, stage, spp, device_balls, poses, device_cache);
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
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            image[static_cast<size_t>(y) * w + x] = render_pixel(x, y, w, h, stage, spp, prepared, poses,
                                                               cache.empty()?-1.f:cache[size_t(y)*w+x]);
    const double render_ms = elapsed_ms(render_start), d2h_ms = 0.0;
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
              << time << ',' << base_seed << ',' << std::fixed << std::setprecision(3) << render_ms << ','
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
