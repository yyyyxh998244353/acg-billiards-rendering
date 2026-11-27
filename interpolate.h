#ifndef ACG_INTERPOLATE_H_INCLUDED
#define ACG_INTERPOLATE_H_INCLUDED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(_OPENMP) && !defined(__CUDACC__)
#include <omp.h>
#endif

// Host-only, single-frame cache. A scalar holds visibility. RGB is the linear direct diffuse factor for
// unit albedo, not shaded radiance. At the destination multiply by that pixel's
// own diffuse albedo, then add its separately evaluated specular, emission and
// indirect terms. Never feed the cache a tonemapped pixel or a whole path sum.
namespace acg_interpolate {

using RGB = std::array<float, 3>;

struct Guide {
    // Opt in only confident, smooth diffuse receivers. Keep misses, mixed
    // coverage/silhouettes, glossy balls, glass, alpha, contact shadows and
    // known fast penumbrae out. A center probe cannot detect every thin shadow.
    bool smooth_diffuse = false;
    int object = -1;    // stable geometry/receiver ID, not just its material
    int material = -1;  // material region; NOT the local texture/albedo value
    float depth = 0;    // positive, same camera-space length units throughout
    std::array<float, 3> normal{}; // unit length, common coordinate frame
};

struct Tolerance { float absolute, relative; };
struct Settings {
    int spacing = 4; // integer pixel-center spacing; clip last cell to image
    float max_depth_delta = .002f; // metres for the billiard scene
    float max_normal_delta = .04f; // chord distance between unit normals
    Tolerance corner_spread{.005f, .05f};
    Tolerance probe_error{.002f, .02f};
};

template<class Light>
struct GridSample { Guide guide; Light light; };
using Sample = GridSample<RGB>;
// Order: top left, top right, bottom left, bottom right.
template<class Light>
using GridCorners = std::array<const GridSample<Light>*, 4>;
using Corners = GridCorners<RGB>;

enum class Source { anchor, probe, fallback, interpolated };
struct Counters {
    // Disjoint pixel classes. Their sum is width*height.
    // Reuse fraction = interpolated / all pixels.
    // In the full-output mode, a probe is emitted at its measured value even
    // when validation fails. Shared anchors and probes are measured once.
    std::uint64_t anchors = 0, probes = 0, fallbacks = 0, interpolated = 0;
    // Cells with at least one non-anchor pixel. Corner rejects skip the probe;
    // probe rejects invalidate the entire cell before any prediction is used.
    std::uint64_t classified_cells = 0, corner_rejects = 0, probe_rejects = 0;
};

inline bool eligible(const Guide& g) {
    if (!g.smooth_diffuse || g.object < 0 || g.material < 0 ||
        !(g.depth > 0) || !std::isfinite(g.depth)) return false;
    const auto& n = g.normal;
    const float n2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
    return std::abs(n2 - 1.f) <= .001f; // also fails closed for NaN/Inf
}

namespace detail {
inline bool nonnegative(float x) { return x >= 0 && std::isfinite(x); }
inline bool valid(Tolerance t) { return nonnegative(t.absolute) && nonnegative(t.relative); }

#if defined(_OPENMP) && !defined(__CUDACC__)
inline void merge_counts(Counters& into, const Counters& from) {
    into.anchors += from.anchors;
    into.probes += from.probes;
    into.fallbacks += from.fallbacks;
    into.interpolated += from.interpolated;
    into.classified_cells += from.classified_cells;
    into.corner_rejects += from.corner_rejects;
    into.probe_rejects += from.probe_rejects;
}
#endif

// Both guides already passed eligible(). Check all pairs, not just each one
// against a representative: opposite corners can straddle a discontinuity.
inline bool compatible(const Guide& a, const Guide& b, const Settings& s) {
    if (a.object != b.object || a.material != b.material ||
        std::abs(a.depth - b.depth) > s.max_depth_delta) return false;
    const float x = a.normal[0]-b.normal[0], y = a.normal[1]-b.normal[1], z = a.normal[2]-b.normal[2];
    return x*x + y*y + z*z <= s.max_normal_delta*s.max_normal_delta;
}

// Callers have already checked both guides for eligibility.
inline bool same_receiver(const Guide& a, const Guide& b) {
    return a.object == b.object && a.material == b.material && a.depth == b.depth && a.normal == b.normal;
}

template<class Light>
inline bool fits(const Guide& g, const GridCorners<Light>& c, const Settings& s) {
    if (!eligible(g)) return false;
    for (const auto* p : c) if (!compatible(g, p->guide, s)) return false;
    return true;
}

template<class Light>
inline Light bilinear(const GridCorners<Light>& c, double u, double v) {
    const double w[] = {(1-u)*(1-v), u*(1-v), (1-u)*v, u*v};
    const auto blend = [&](float a, float b, float c, float d) {
        return static_cast<float>(w[0]*a + w[1]*b + w[2]*c + w[3]*d);
    };
    if constexpr (std::is_same_v<Light,float>) {
        return blend(c[0]->light,c[1]->light,c[2]->light,c[3]->light);
    } else {
        RGB result;
        for (std::size_t k = 0; k < result.size(); ++k)
            result[k] = blend(c[0]->light[k],c[1]->light[k],c[2]->light[k],c[3]->light[k]);
        return result;
    }
}
} // namespace detail

// Conservative residual per component with an absolute floor in
// linear light units, relative allowance measured against the darker sample.
inline bool validate(float a, float b, Tolerance t) {
    return detail::nonnegative(a) && detail::nonnegative(b) &&
           std::abs(a-b) <= t.absolute + double(t.relative)*std::min(a,b);
}

inline bool validate(const RGB& a, const RGB& b, Tolerance t) {
    for (std::size_t k = 0; k < a.size(); ++k)
        if (!validate(a[k],b[k],t)) return false;
    return true;
}

// True means "worth probing", not permission to interpolate every pixel.
template<class Light=RGB>
inline bool classify_cell(const GridCorners<Light>& c, const Settings& s) {
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (!c[i] || !eligible(c[i]->guide)) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (!detail::compatible(c[i]->guide, c[j]->guide, s) ||
                !validate(c[i]->light, c[j]->light, s.corner_spread)) return false;
    }
    return true;
}

#if defined(_OPENMP) && !defined(__CUDACC__)
#pragma omp declare reduction(grid_counts : Counters : detail::merge_counts(omp_out,omp_in)) initializer(omp_priv = Counters{})
#endif

// Run once per fixed camera/pose/sample domain (no temporal or lens reuse).
//   guide(x,y) -> Guide, or a derived type carrying caller-owned hit data
//   exact(x,y,guide) -> float or RGB, an unfiltered linear light measurement
//   emit(x,y,guide,light,Source) consumes the selected samples.
// With EmitExact=true, guide and emit run once per pixel. With EmitExact=false,
// emit receives only interpolated samples, and unneeded guide/exact calls are
// skipped. Callbacks must agree on the primary hit and use coordinate-keyed
// sampling, independent of evaluation order. Emission is not scanline ordered.
// Non-finite/negative measurements never authorize reuse.
// exact measurements themselves are passed through, not repaired by filtering.
// Serial mode holds two anchor rows. Parallel mode processes bounded batches
// of anchor rows and requires thread-safe, coordinate-keyed, non-throwing callbacks.
template<bool EmitExact=true, bool Parallel=false, class Lookup, class Exact, class Emit>
Counters fixed_grid(int width, int height, const Settings& s, Lookup&& guide, Exact&& exact, Emit&& emit) {
    static_assert(!Parallel || !EmitExact,"parallel grid requires interpolated-only output");
    if (width < 1 || height < 1 || s.spacing < 1 || !detail::nonnegative(s.max_depth_delta) ||
        !detail::nonnegative(s.max_normal_delta) || !detail::valid(s.corner_spread) || !detail::valid(s.probe_error))
        throw std::invalid_argument("invalid interpolation grid or tolerance");
    Counters count;
    auto measure = [&](int x, int y, Source source) {
        const auto& g = guide(x,y);
        using Result = std::decay_t<decltype(exact(x,y,g))>;
        using Light = std::conditional_t<std::is_same_v<Result,float>,float,RGB>;
        const Light light = exact(x,y,g);
        if constexpr (EmitExact) emit(x,y,g,light,source);
        return GridSample<Light>{g,light};
    };
    using Anchor = decltype(measure(0,0,Source::anchor));
    using Light = decltype(Anchor::light);

    // Unit spacing is the exact reference. With no 2D cell, measure the
    // regular grid anchors and fall back everywhere between them.
    if (s.spacing == 1 || width == 1 || height == 1) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const bool anchor = (x % s.spacing == 0 || x == width-1) && (y % s.spacing == 0 || y == height-1);
            if constexpr (EmitExact) measure(x, y, anchor ? Source::anchor : Source::fallback);
            if (anchor) ++count.anchors; else ++count.fallbacks;
        }
        return count;
    }

    const int nx = 2 + (width-2)/s.spacing;
    auto grid_x = [&](int i) { return i == nx-1 ? width-1 : i*s.spacing; };
    auto process_band = [&](int y0, int y1, const Anchor* top, const Anchor* bottom, Counters& count) {
        for (int i = 0; i < nx-1; ++i) {
            const int x0 = grid_x(i), x1 = grid_x(i+1);
            if (x1-x0 == 1 && y1-y0 == 1) continue; // corners are the only pixels
            const GridCorners<Light> c{&top[i], &top[i+1], &bottom[i], &bottom[i+1]};
            ++count.classified_cells;
            const int dx = x1-x0, dy = y1-y0;
            const bool tested = classify_cell(c, s);
            const Guide& first = c[0]->guide;
            const bool uniform_corners = tested && detail::same_receiver(first,c[1]->guide) &&
                detail::same_receiver(first,c[2]->guide) && detail::same_receiver(first,c[3]->guide);
            const bool fully_lit = [&] {
                if constexpr (std::is_same_v<Light,float>) {
                    // Small power-of-two spans give exact dyadic weights.
                    const auto exact_span = [](int span) { return span<=64 && (span & (span-1))==0; };
                    return tested && exact_span(dx) && exact_span(dy) &&
                           c[0]->light==1.f && c[1]->light==1.f && c[2]->light==1.f && c[3]->light==1.f;
                }
                return false;
            }();
            const double du = 1.0/dx, dv = 1.0/dy;
            const auto predict = [&](int x, int y) {
                if constexpr (std::is_same_v<Light,float>)
                    if (fully_lit) return 1.f;
                return detail::bilinear(c, (x-x0)*du, (y-y0)*dv);
            };
            const int px = x0+(x1-x0)/2, py = y0+(y1-y0)/2;
            bool use = false;
            Anchor probe{};
            if (tested) {
                probe = measure(px, py, Source::probe);
                ++count.probes;
                const bool compatible_probe = uniform_corners
                    ? eligible(probe.guide) && detail::compatible(probe.guide,first,s)
                    : detail::fits(probe.guide,c,s);
                use = compatible_probe && validate(probe.light, predict(px, py), s.probe_error);
                if (!use) ++count.probe_rejects;
            } else ++count.corner_rejects;
            const bool uniform_cell = use && uniform_corners && detail::same_receiver(first,probe.guide);

            // Half-open ownership on shared edges; include the last column/row
            // of the image. Skip previously emitted grid points and center probe.
            const int end_x = x1 + (x1 == width-1), end_y = y1 + (y1 == height-1);
            if constexpr (!EmitExact) {
                if (!use) {
                    // Only the final row and column own their far-edge anchors.
                    const int anchors = (1+(x1==width-1)) * (1+(y1==height-1));
                    const auto pixels = std::uint64_t(end_x-x0)*(end_y-y0);
                    count.fallbacks += pixels - anchors - tested; // A tested cell already measured its probe.
                    continue;
                }
            }
            for (int y = y0; y < end_y; ++y) for (int x = x0; x < end_x; ++x) {
                if (((x == x0 || x == x1) && (y == y0 || y == y1)) || (tested && x == px && y == py)) continue;
                const auto& g = guide(x,y);
                if (use && (uniform_cell ? acg_interpolate::eligible(g) && detail::compatible(g,first,s) :
                    detail::fits(g,c,s) && detail::compatible(g,probe.guide,s))) {
                    emit(x, y, g, predict(x,y), Source::interpolated);
                    ++count.interpolated;
                } else {
                    if constexpr (EmitExact) emit(x, y, g, exact(x,y,g), Source::fallback);
                    ++count.fallbacks;
                }
            }
        }
    };
#if defined(_OPENMP) && !defined(__CUDACC__)
    if constexpr (Parallel) {
        constexpr int bands_per_batch = 32;
        const int ny = 2 + (height-2)/s.spacing;
        if (std::uint64_t(width)*height >= 32*1024 && ny>4 && omp_get_max_threads()>1) {
            const auto grid_y = [&](int j) { return j==ny-1 ? height-1 : j*s.spacing; };
            const int row_count = std::min(ny,bands_per_batch+1);
            const auto slots = std::uint64_t(nx)*row_count;
            std::vector<Anchor> anchors;
            if (slots>anchors.max_size()) throw std::length_error("interpolation anchor buffer too large");
            anchors.resize(static_cast<std::size_t>(slots));
            const auto anchor_row = [&](int j) { return anchors.data()+std::size_t(j%row_count)*nx; };
            #pragma omp parallel reduction(grid_counts:count)
            {
                for (int begin=0; begin<ny-1;) {
                    const int end = begin+std::min(bands_per_batch,ny-1-begin);
                    // The previous batch's final row is this batch's first.
                    #pragma omp for schedule(static,1)
                    for (int j=begin+(begin!=0); j<=end; ++j) {
                        Anchor* const row = anchor_row(j);
                        const int y = grid_y(j);
                        for (int i=0; i<nx; ++i) row[i] = measure(grid_x(i),y,Source::anchor);
                    }
                    // Both adjacent rows stay live until all cells finish.
                    #pragma omp for schedule(static,1)
                    for (int j=begin; j<end; ++j)
                        process_band(grid_y(j),grid_y(j+1),anchor_row(j),anchor_row(j+1),count);
                    begin=end;
                }
            }
            count.anchors = std::uint64_t(nx)*ny;
            return count;
        }
    }
#endif
    std::vector<Anchor> top, bottom;
    top.reserve(nx); bottom.reserve(nx);
    auto fill_row = [&](std::vector<Anchor>& row, int y) {
        row.clear();
        for (int i = 0; i < nx; ++i) row.push_back(measure(grid_x(i), y, Source::anchor));
        count.anchors += nx;
    };
    fill_row(top, 0);
    for (int y0 = 0; y0 < height-1;) {
        const int y1 = y0 + std::min(s.spacing, height-1-y0);
        fill_row(bottom, y1);
        process_band(y0,y1,top.data(),bottom.data(),count);
        top.swap(bottom);
        y0 = y1;
    }
    return count;
}

} // namespace acg_interpolate
#endif
