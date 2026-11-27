#ifndef ACG_FELT_PYRAMID_H
#define ACG_FELT_PYRAMID_H

#include "scene.h"

namespace acg_felt {
using namespace acg_scene_detail;

constexpr int pitch=8;
constexpr Vec invalid={-1,0,0};
constexpr float contact_pad=.085f;

struct Grid {
    const Vec* values;
    int width,height;
};

// Each node certifies a square extending one fine pitch in every direction.
// Four certified corners therefore cover the full support of either level.
ACG_HD inline bool anchor(int x,int y,int w,int h,const Balls& balls,Vec& p) {
    if(!overhead_slate_clear) return false;
    const float aspect=float(w)/h, halfheight=hi(.77f,1.40f/aspect);
    p={(2*(x+.5f)/w-1)*aspect*halfheight,0,(2*(y+.5f)/h-1)*halfheight};
    // Full pixel jitter, a conservative 4 mm thin-lens bound, and roundoff.
    const float support=(pitch+.5f)*2*halfheight/h+.004f+4*eps;
    if(fabsf(p.x)+support>=table_clear_x || fabsf(p.z)+support>=table_clear_z) return false;
    for(const Vec c:balls.c)
        if(fabsf(c.x-p.x)<=support+contact_pad && fabsf(c.z-p.z)<=support+contact_pad) return false;

    // Expanding occluders in XZ by support bounds the union of the beams
    // from every receiver in this square. Reject all penumbrae, including
    // those from the static room, rather than trusting a center-ray test.
    const Vec org=p+Vec{0,eps,0}, pad={support,0,support};
    const double height=double(light_y)-org.y;
    const SoftBeam beam={org,height,(-double(light_hx)-org.x)/height,(double(light_hx)-org.x)/height,
        (double(light_z)-light_hz-org.z)/height,(double(light_z)+light_hz-org.z)/height};
    for(const Vec c:balls.c)
        if(soft_beam_overlaps(beam,{c-grey(radius)-pad,c+grey(radius)+pad})) return false;
    const StaticScene& scene=room();
    for(int i=0;i<scene.size;++i) {
        const Box box=scene.p[i].box;
        if(soft_beam_overlaps(beam,{box.a-pad,box.b+pad})) return false;
    }
    return true;
}

// Tent-filter the fine anchors around each coincident coarse node. Never
// fill holes by averaging only the valid neighbors of an occluder.
ACG_HD inline Vec coarse_node(Grid fine,int x,int y) {
    x*=2; y*=2;
    if(x<=0 || y<=0 || x+1>=fine.width || y+1>=fine.height) return invalid;
    Vec sum{};
    for(int j=-1;j<=1;++j) for(int i=-1;i<=1;++i) {
        const Vec v=fine.values[(y+j)*fine.width+x+i];
        if(!(v.x>=0)) return invalid;
        sum=sum+v*(float((i==0?2:1)*(j==0?2:1))*(1.f/16));
    }
    return sum;
}

ACG_HD inline const Vec* valid_cell(Grid grid,int x,int y,int step) {
    const int u=x/step,v=y/step;
    if(u+1>=grid.width || v+1>=grid.height) return nullptr;
    const Vec* row=grid.values+v*grid.width+u;
    if(!(row[0].x>=0 && row[1].x>=0 && row[grid.width].x>=0 && row[grid.width+1].x>=0)) return nullptr;
    return row;
}

ACG_HD inline Vec bilinear(Grid grid,int x,int y,int step) {
    const Vec* row=valid_cell(grid,x,y,step);
    if(!row) return invalid;
    const Vec a=row[0],b=row[1],c=row[grid.width],d=row[grid.width+1];
    const float fx=float(x%step)/step,fy=float(y%step)/step;
    return mix(mix(a,b,fx),mix(c,d,fx),fy);
}

// Blending at nodes makes the coarse-to-fine transition continuous over
// neighboring fine cells, even when one of them has no valid coarse support.
ACG_HD inline Vec blend_node(Vec fine,Grid coarse,int x,int y) {
    if(!(fine.x>=0)) return invalid;
    const Vec broad=bilinear(coarse,x*pitch,y*pitch,2*pitch);
    if(broad.x<0) return fine;
    return mix(fine,broad,.75f);
}
} // namespace acg_felt
#endif
