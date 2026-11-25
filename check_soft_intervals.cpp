// Compile from the repository root: g++ -O2 -std=c++17 check_soft_intervals.cpp -o check_soft && ./check_soft
#include "scene.h"
#include <cstdio>
#include <initializer_list>

int main() {
    using namespace acg_scene_detail;
    constexpr int samples=20000;
    const Vec p={0,0,0};
    {
        // Center ray intersects a sphere lying halfway to the emitter; a
        // narrow slice lies entirely inside its projected silhouette.
        SoftEmitter lamp={1,0,.01f,.01f};
        Vec centers[16];
        for(auto& center:centers) center={9,radius,9};
        centers[0]={0,.5f,0};
        SoftSlice slice;
        if(!soft_slice(p,centers,0,slice,lamp) || slice.count!=0 || slice.geometry!=0) {
            std::fputs("Fully blocked slice emitted light\n",stderr);
            return 1;
        }
        // The X/Y plane touches the sphere at one point: zero blocked length.
        constexpr float tangent_radius=.125f;
        centers[0]={0,.5f,tangent_radius};
        if(!soft_slice(p,centers,0,slice,lamp,tangent_radius) || slice.count!=1 ||
           slice.visible[0].a!=-double(lamp.hx) || slice.visible[0].b!=double(lamp.hx)) {
            std::fputs("Plane tangency removed a visible interval\n",stderr);
            return 1;
        }
    }
    for (int setup=0;setup<4;++setup) {
        Vec centers[16];
        for(auto& center:centers) center={9,radius,9};
        if(setup) centers[0]={0,radius,setup==3?0.f:-.05f};
        if(setup>=2) centers[1]={.035f,radius,-.055f};
        for(float width:{.12f,light_hx}) for(int s=0;s<9;++s) {
            SoftEmitter lamp=soft_lamp;
            lamp.hx=width;
            float z=lamp.z-lamp.hz+2*lamp.hz*(s+.5f)/9;
            SoftSlice slice;
            if(!soft_slice(p,centers,z,slice,lamp)) return 1;
            double numeric=0;
            for(int i=0;i<samples;++i) {
                float x=-width+2*width*(i+.5f)/samples;
                Vec d={x,lamp.y,z};
                float length=sqrtf(dot(d,d)), t;
                bool visible=true;
                for(auto center:centers) if(sphere_t({p,d*(1/length)},center,radius,0,length,t)) {
                    visible=false;
                    break;
                }
                if(visible!=soft_visible(slice,x)) {
                    std::fprintf(stderr,"Interval mismatch: setup %d, width %g, sample %d\n",setup,width,i);
                    return 1;
                }
                if(visible) numeric+=double(lamp.y)*lamp.y/pow(double(length),4);
            }
            if(fabs(numeric*2*width/samples-slice.geometry)>.00015) {
                std::fprintf(stderr,"Integral mismatch: setup %d, width %g, slice %d\n",setup,width,s);
                return 1;
            }
        }
    }
    std::puts("72 quadrature slices plus full blockage and plane tangency passed.");
}
