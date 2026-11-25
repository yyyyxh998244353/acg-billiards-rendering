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
#ifdef __CUDACC__
__constant__ float device_balls[64 * 32];
#endif

namespace {
using Clock = std::chrono::steady_clock;
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

RENDER_HD inline RGB8 render_pixel(int x, int y, int w, int h, int stage, int spp, float time, const float* balls, unsigned poses) {
    // No dependency on launch order, stage, spp budget or time. Matching pixels
    // share the same prefix of random samples in all comparison renders.
    const u32 pixel_key = mix_bits(base_seed ^ static_cast<u32>(y)) ^ static_cast<u32>(x);
    const u32 first = mix_bits(pixel_key);
    float r = 0, g = 0, b = 0;
    for (int s = 0; s < spp; ++s) {
        const u32 seed = mix_bits(first + static_cast<u32>(s));
        const unsigned pose = (static_cast<std::uint64_t>(mix_bits(seed ^ 0xba7c19u)) * poses) >> 32;
        const Vec v = sample_pixel(x, y, w, h, stage, seed ? seed : base_seed, time, balls + 32 * pose);
        r += v.x; g += v.y; b += v.z;
    }
    const float inv = 1.0f / spp;
    return {display_byte(r * inv), display_byte(g * inv), display_byte(b * inv)};
}

#ifdef __CUDACC__
__global__ void render_pixels(RGB8* pixels, size_t count, int w, int h, int stage, int spp, float time, unsigned poses) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
        pixels[i] = render_pixel(static_cast<int>(i % w), static_cast<int>(i / w), w, h, stage, spp, time, device_balls, poses);
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
    float balls[64 * 32];
    const unsigned poses = argc - 7;
    for (unsigned i = 0; i < poses; ++i) {
        std::ifstream snapshot(argv[7 + i]);
        for (unsigned j = 0; j < 32; ++j) {
            float& value = balls[32 * i + j];
            if (!(snapshot >> value) || !std::isfinite(value)) throw std::runtime_error("invalid physics snapshot");
        }
        std::string extra;
        if (snapshot >> extra) throw std::runtime_error("extra physics snapshot values");
    }
    const auto total_start = Clock::now();
    const std::uint64_t count64 = static_cast<std::uint64_t>(w) * h;
    if (count64 > std::numeric_limits<size_t>::max() / sizeof(RGB8) ||
        count64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(RGB8))
        throw std::length_error("image byte count exceeds platform limit");
    const size_t count = static_cast<size_t>(count64), bytes = count * sizeof(RGB8);
    std::unique_ptr<RGB8[]> image(new RGB8[count]);
#ifdef __CUDACC__
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "CUDA device 0: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    CUDA_CHECK(cudaMemcpyToSymbol(device_balls, balls, poses * 32 * sizeof(float)));
    RGB8* device = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    constexpr unsigned threads = 128;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count - 1) / threads + 1, 65535));
    CUDA_CHECK(cudaEventRecord(begin));
    render_pixels<<<blocks, threads>>>(device, count, w, h, stage, spp, time, poses);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float render_ms;
    CUDA_CHECK(cudaEventElapsedTime(&render_ms, begin, end));
    const auto copy_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(image.get(), device, bytes, cudaMemcpyDeviceToHost));
    const double d2h_ms = elapsed_ms(copy_start);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaFree(device));
    constexpr const char* backend = "cuda";
#else
    const auto render_start = Clock::now();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            image[static_cast<size_t>(y) * w + x] = render_pixel(x, y, w, h, stage, spp, time, balls, poses);
    const double render_ms = elapsed_ms(render_start), d2h_ms = 0.0;
    constexpr const char* backend = "cpu";
#endif
    const auto io_start = Clock::now();
    std::ofstream out(argv[5], std::ios::binary);
    out << "P6\n" << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.get()), static_cast<std::streamsize>(bytes));
    out.close();
    if (!out) throw std::runtime_error(std::string("failed writing ") + argv[5]);
    const double io_ms = elapsed_ms(io_start), total_ms = elapsed_ms(total_start);
    // render_ms = one kernel (sampling + tonemap) on CUDA, render loop on CPU.
    // total_ms = cold invocation from buffer setup through PPM close, not FPS.
    std::cout << "backend,stage,width,height,spp,time,base_seed,render_ms,d2h_wall_ms,ppm_wall_ms,total_wall_ms\n"
              << backend << ',' << stage << ',' << w << ',' << h << ',' << spp << ',' << std::setprecision(9)
              << time << ',' << base_seed << ',' << std::fixed << std::setprecision(3) << render_ms << ','
              << d2h_ms << ',' << io_ms << ',' << total_ms << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "render: " << e.what() << '\n';
    return 1;
}
