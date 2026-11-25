#ifndef ACG_INTERPOLATE_H_INCLUDED
#define ACG_INTERPOLATE_H_INCLUDED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Host-only, single-frame cache. RGB is the LINEAR direct diffuse factor for
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

struct Sample { Guide guide; RGB light; };
// Order: top left, top right, bottom left, bottom right.
using Corners = std::array<const Sample*, 4>;

enum class Source { anchor, probe, fallback, interpolated };
struct Counters {
    // Disjoint output pixels. Their sum is width*height; exact() calls are
    // anchors + probes + fallbacks. Reuse fraction = interpolated / all pixels.
    // A probe is emitted at its measured value, even when validation fails;
    // neither shared anchors nor failed probes are charged a second time.
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

// Both guides already passed eligible(). Check all pairs, not just each one
// against a representative: opposite corners can straddle a discontinuity.
inline bool compatible(const Guide& a, const Guide& b, const Settings& s) {
    if (a.object != b.object || a.material != b.material ||
        std::abs(a.depth - b.depth) > s.max_depth_delta) return false;
    const float x = a.normal[0]-b.normal[0], y = a.normal[1]-b.normal[1], z = a.normal[2]-b.normal[2];
    return x*x + y*y + z*z <= s.max_normal_delta*s.max_normal_delta;
}

inline bool fits(const Guide& g, const Corners& c, const Settings& s) {
    if (!eligible(g)) return false;
    for (const Sample* p : c) if (!compatible(g, p->guide, s)) return false;
    return true;
}

inline RGB bilinear(const Corners& c, double u, double v) {
    const double w[] = {(1-u)*(1-v), u*(1-v), (1-u)*v, u*v};
    RGB result;
    for (std::size_t k = 0; k < result.size(); ++k)
        result[k] = static_cast<float>(w[0]*c[0]->light[k] + w[1]*c[1]->light[k] +
                                       w[2]*c[2]->light[k] + w[3]*c[3]->light[k]);
    return result;
}
} // namespace detail

// Conservative per-channel residual (not just luminance): absolute floor in
// linear light units, relative allowance measured against the darker sample.
inline bool validate(const RGB& a, const RGB& b, Tolerance t) {
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (!detail::nonnegative(a[k]) || !detail::nonnegative(b[k]) ||
            !(std::abs(a[k]-b[k]) <= t.absolute + double(t.relative)*std::min(a[k], b[k])))
            return false;
    }
    return true;
}

// True means "worth probing", not permission to interpolate every pixel.
inline bool classify_cell(const Corners& c, const Settings& s) {
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (!c[i] || !eligible(c[i]->guide)) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (!detail::compatible(c[i]->guide, c[j]->guide, s) ||
                !validate(c[i]->light, c[j]->light, s.corner_spread)) return false;
    }
    return true;
}

// Run once per fixed camera/pose/sample domain (no temporal or lens reuse).
//   guide(x,y) -> Guide, or a derived type carrying caller-owned hit data
//   exact(x,y,guide) -> RGB, the ordinary unfiltered unit-albedo diffuse term
//   emit(x,y,guide,RGB,Source) consumes the result, including on unsafe hits.
// guide and emit run once per pixel. Callbacks must agree on the primary hit;
// use coordinate-keyed sampling, not an order-dependent RNG. Emission is not
// scanline ordered. Non-finite/negative measurements never authorize reuse;
// exact measurements themselves are passed through, not repaired by filtering.
// Memory: two anchor rows; no full-resolution buffer or persistent scene state.
template<class Lookup, class Exact, class Emit>
Counters fixed_grid(int width, int height, const Settings& s, Lookup&& guide, Exact&& exact, Emit&& emit) {
    if (width < 1 || height < 1 || s.spacing < 1 || !detail::nonnegative(s.max_depth_delta) ||
        !detail::nonnegative(s.max_normal_delta) || !detail::valid(s.corner_spread) || !detail::valid(s.probe_error))
        throw std::invalid_argument("invalid interpolation grid or tolerance");
    Counters count;
    auto measure = [&](int x, int y, Source source) -> Sample {
        const auto& g = guide(x,y);
        const RGB light = exact(x,y,g);
        emit(x,y,g,light,source);
        return {g, light};
    };

    // Unit spacing is the exact reference. With no 2D cell, measure the
    // regular grid anchors and fall back everywhere between them.
    if (s.spacing == 1 || width == 1 || height == 1) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const bool anchor = (x % s.spacing == 0 || x == width-1) && (y % s.spacing == 0 || y == height-1);
            measure(x, y, anchor ? Source::anchor : Source::fallback);
            if (anchor) ++count.anchors; else ++count.fallbacks;
        }
        return count;
    }

    const int nx = 2 + (width-2)/s.spacing;
    auto grid_x = [&](int i) { return i == nx-1 ? width-1 : i*s.spacing; };
    std::vector<Sample> top, bottom;
    top.reserve(nx); bottom.reserve(nx);
    auto fill_row = [&](std::vector<Sample>& row, int y) {
        row.clear();
        for (int i = 0; i < nx; ++i) row.push_back(measure(grid_x(i), y, Source::anchor));
        count.anchors += nx;
    };
    fill_row(top, 0);
    for (int y0 = 0; y0 < height-1;) {
        const int y1 = y0 + std::min(s.spacing, height-1-y0);
        fill_row(bottom, y1);
        for (int i = 0; i < nx-1; ++i) {
            const int x0 = grid_x(i), x1 = grid_x(i+1);
            if (x1-x0 == 1 && y1-y0 == 1) continue; // corners are the only pixels
            const Corners c{&top[i], &top[i+1], &bottom[i], &bottom[i+1]};
            ++count.classified_cells;
            const double du = 1.0/(x1-x0), dv = 1.0/(y1-y0);
            auto predict = [&](int x, int y) { return detail::bilinear(c, (x-x0)*du, (y-y0)*dv); };
            const int px = x0+(x1-x0)/2, py = y0+(y1-y0)/2;
            const bool tested = classify_cell(c, s);
            bool use = false;
            Sample probe{};
            if (tested) {
                probe = measure(px, py, Source::probe);
                ++count.probes;
                use = detail::fits(probe.guide, c, s) && validate(probe.light, predict(px, py), s.probe_error);
                if (!use) ++count.probe_rejects;
            } else ++count.corner_rejects;

            // Half-open ownership on shared edges; include the last column/row
            // of the image. Skip previously emitted grid points and center probe.
            const int end_x = x1 + (x1 == width-1), end_y = y1 + (y1 == height-1);
            for (int y = y0; y < end_y; ++y) for (int x = x0; x < end_x; ++x) {
                if (((x == x0 || x == x1) && (y == y0 || y == y1)) || (tested && x == px && y == py)) continue;
                const auto& g = guide(x,y);
                if (use && detail::fits(g, c, s) && detail::compatible(g, probe.guide, s)) {
                    emit(x, y, g, predict(x,y), Source::interpolated);
                    ++count.interpolated;
                } else {
                    emit(x, y, g, exact(x,y,g), Source::fallback);
                    ++count.fallbacks;
                }
            }
        }
        top.swap(bottom);
        y0 = y1;
    }
    return count;
}

} // namespace acg_interpolate
#endif
