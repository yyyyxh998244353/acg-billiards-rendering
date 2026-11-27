#ifndef ACG_PRIMARY_SAMPLING_H
#define ACG_PRIMARY_SAMPLING_H

// Included in acg_scene_detail after hash32(). Each projection has its own
// digital shift; changing the path count never moves an existing sample.
struct FirstSample {
    uint32_t sample=~0u,key=0;
    bool sobol=false;
};

enum PrimaryDimension { RasterX, RasterY, EmitterX, EmitterZ, Lobe, DirectionU, DirectionV, PrimaryDimensions };
struct PrimaryDirections { uint32_t values[PrimaryDimensions][32]; };

constexpr PrimaryDirections make_primary_directions() {
    PrimaryDirections ds{};
    // The first seven dimensions of the Joe–Kuo direction table.
    constexpr int degrees[]={1,2,3,3,4,4};
    constexpr uint32_t coefficients[]={0,1,1,2,1,4};
    constexpr uint32_t starts[][4]={{1},{1,3},{1,3,1},{1,1,1},{1,1,3,3},{1,3,5,13}};
    for(int bit=0;bit<32;++bit) ds.values[0][bit]=uint32_t(1)<<(31-bit);
    for(int dim=1;dim<PrimaryDimensions;++dim) {
        const int degree=degrees[dim-1];
        auto& v=ds.values[dim];
        for(int i=0;i<degree;++i) v[i]=starts[dim-1][i]<<(31-i);
        for(int i=degree;i<32;++i) {
            v[i]=v[i-degree]^(v[i-degree]>>degree);
            for(int k=1;k<degree;++k)
                if((coefficients[dim-1]>>(degree-1-k))&1u) v[i]^=v[i-k];
        }
    }
    return ds;
}

static constexpr PrimaryDirections primary_host=make_primary_directions();
#if defined(__CUDACC__) || defined(__CUDA__)
static __device__ __constant__ PrimaryDirections primary_device=make_primary_directions();
#endif

ACG_HD inline float primary_sobol(FirstSample first,PrimaryDimension dimension) {
#if defined(__CUDA_ARCH__)
    const uint32_t* directions=primary_device.values[dimension];
#else
    const uint32_t* directions=primary_host.values[dimension];
#endif
    uint32_t index=first.sample^(first.sample>>1);
    uint32_t word=hash32(first.key^(0x9e3779b9u*(uint32_t(dimension)+1)));
    while(index) {
#if defined(__CUDA_ARCH__)
        const int bit=__ffs(static_cast<int>(index))-1;
#elif defined(_MSC_VER)
        unsigned long bit;
        _BitScanForward(&bit,index);
#else
        const int bit=__builtin_ctz(index);
#endif
        word^=directions[bit];
        index&=index-1;
    }
    return float(word>>8)*0x1p-24f;
}
#endif
