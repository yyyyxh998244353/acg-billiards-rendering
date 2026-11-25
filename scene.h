#ifndef ACG_SCENE_H_INCLUDED
#define ACG_SCENE_H_INCLUDED

#include <math.h>
#include <stdint.h>

#ifndef ACG_HD
#  if defined(__CUDACC__) || defined(__CUDA__)
#    define ACG_HD __host__ __device__
#  else
#    define ACG_HD
#  endif
#endif

struct Vec { float x, y, z; };

namespace acg_scene_detail {
constexpr float pi = 3.14159265358979323846f;
constexpr float radius = .028575f; // metres; world Y is up
constexpr float eps = .000012f;
constexpr float far_t = 1.e19f;

ACG_HD constexpr Vec grey(float a) { return {a,a,a}; }
ACG_HD constexpr Vec operator+(Vec a, Vec b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
ACG_HD constexpr Vec operator-(Vec a, Vec b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
ACG_HD constexpr Vec operator-(Vec a) { return {-a.x,-a.y,-a.z}; }
ACG_HD constexpr Vec operator*(Vec a, float s) { return {a.x*s,a.y*s,a.z*s}; }
ACG_HD constexpr Vec operator*(Vec a, Vec b) { return {a.x*b.x,a.y*b.y,a.z*b.z}; }
ACG_HD constexpr float dot(Vec a, Vec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
ACG_HD constexpr Vec cross(Vec a, Vec b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
ACG_HD constexpr float lo(float a,float b) { return a<b?a:b; }
ACG_HD constexpr float hi(float a,float b) { return a>b?a:b; }
ACG_HD constexpr float clamp(float x,float a,float b) { return lo(hi(x,a),b); }
ACG_HD constexpr float peak(Vec v) { return hi(v.x,hi(v.y,v.z)); }
ACG_HD constexpr float axis(Vec a,int i) { return i==0?a.x:(i==1?a.y:a.z); }
ACG_HD inline Vec unit(Vec a) { return a*(1.f/sqrtf(dot(a,a))); }
ACG_HD constexpr Vec mix(Vec a,Vec b,float f) { return a*(1-f)+b*f; }
ACG_HD inline float fract(float x) { return x-floorf(x); }
ACG_HD inline Vec rotate(Vec v,Vec u,float a) {
    float c=cosf(a),s=sinf(a);
    return v*c+cross(u,v)*s+u*(dot(u,v)*(1-c));
}
ACG_HD inline uint32_t hash32(uint32_t a) {
    a^=a>>16; a*=0x7feb352du; a^=a>>15; a*=0x846ca68bu; return a^(a>>16);
}
struct Rng {
    uint32_t s;
    ACG_HD float next() {
        // PCG RXS-M-XS; use exactly 24 bits so conversion cannot round to 1.
        s=s*747796405u+2891336453u;
        uint32_t b=((s>>((s>>28u)+4u))^s)*277803737u;
        return float(((b>>22u)^b)>>8u)*0x1p-24f;
    }
};

struct Ray { Vec o,d; };
struct Box { Vec a,b; };
ACG_HD constexpr Box unite(Box a,Box b) {
    return {{lo(a.a.x,b.a.x),lo(a.a.y,b.a.y),lo(a.a.z,b.a.z)},
            {hi(a.b.x,b.b.x),hi(a.b.y,b.b.y),hi(a.b.z,b.b.z)}};
}
ACG_HD inline float recip(float f) { return copysignf(1.f/hi(fabsf(f),1.e-20f),f); }
ACG_HD inline Vec invdir(Vec d) { return {recip(d.x),recip(d.y),recip(d.z)}; }
ACG_HD inline bool overlaps(Box b,Ray r,Vec inv,float t0,float t1) {
    Vec l=(b.a-r.o)*inv,h=(b.b-r.o)*inv;
    return hi(t0,hi(lo(l.x,h.x),hi(lo(l.y,h.y),lo(l.z,h.z)))) <=
           lo(t1,lo(hi(l.x,h.x),lo(hi(l.y,h.y),hi(l.z,h.z))));
}

enum Surface { Felt, Timber, Brass, Ebony, Leather, Floor, Wall, Ceiling, Glass, Grille, Ball, Lamp };
enum Shape { Block, BoredBlock, CushionX, CushionZ, Well, Globe, Screen };
struct Prim { Box box; Shape shape; Surface mat; int side; };
struct Node { Box box; int first,count,escape; };
// A small, immutable preorder BVH. Construction/sorting happen at compile time,
// not once per thread or per frame. Escape indices make traversal stackless.
struct StaticScene {
    Prim p[96]; Node n[96]; int size,nodes;
    ACG_HD constexpr void add(Vec a,Vec b,Surface m,Shape s=Block,int side=0) {
        p[size++]={{a,b},s,m,side};
    }
    ACG_HD constexpr void build(int first,int count) {
        int k=nodes++; Box box=p[first].box;
        for(int i=1;i<count;++i) box=unite(box,p[first+i].box);
        n[k]={box,first,count,k+1};
        if(count<=3) return;
        Box cb={(p[first].box.a+p[first].box.b)*.5f,(p[first].box.a+p[first].box.b)*.5f};
        for(int i=1;i<count;++i) {
            Vec c=(p[first+i].box.a+p[first+i].box.b)*.5f; cb=unite(cb,{c,c});
        }
        Vec d=cb.b-cb.a; int ax=d.x>d.y?0:1; if(d.z>axis(d,ax)) ax=2;
        for(int i=first+1;i<first+count;++i) {
            Prim v=p[i]; float c=axis(v.box.a+v.box.b,ax); int j=i;
            while(j>first && axis(p[j-1].box.a+p[j-1].box.b,ax)>c) { p[j]=p[j-1]; --j; }
            p[j]=v;
        }
        n[k].count=0; build(first,count/2); build(first+count/2,count-count/2);
        n[k].escape=nodes;
    }
};
constexpr float pocket_x=1.283f,pocket_z=.652f,bore=.069f;
constexpr float pocket_top=.049f,pocket_throat=.056f,pocket_bevel=.025f;
constexpr float pocket_slope=(bore-pocket_throat)/pocket_bevel;
ACG_HD constexpr StaticScene make_room() {
    StaticScene s={};
    s.add({-3.6f,-.85f,-2.65f},{3.6f,-.82f,3.65f},Floor);
    s.add({-3.63f,-.85f,-2.68f},{-3.6f,2.55f,3.65f},Wall);
    s.add({3.6f,-.85f,-2.68f},{3.63f,2.55f,3.65f},Wall);
    s.add({-3.6f,-.85f,-2.68f},{3.6f,2.55f,-2.65f},Wall);
    s.add({-3.6f,-.85f,3.65f},{3.6f,2.55f,3.68f},Wall);
    s.add({-3.63f,2.55f,-2.68f},{3.63f,2.58f,3.68f},Ceiling);
    // Flat slate, rails, cushions and six bored pockets.
    s.add({-1.35f,-.11f,-.716f},{1.35f,0,.716f},Felt,BoredBlock);
    for(int side=-1;side<=1;side+=2) {
        float z=side*.739f,x=side*1.374f;
        s.add({-1.422f,-.115f,z-.048f},{1.422f,.048f,z+.048f},Timber,BoredBlock);
        s.add({x-.048f,-.115f,-.691f},{x+.048f,.048f,.691f},Timber,BoredBlock);
        float qz=side*.667f,qx=side*1.302f;
        for(int half=-1;half<=1;half+=2) {
            float cx=half*.638f;
            s.add({cx-.581f,0,qz-.032f},{cx+.581f,.044f,qz+.032f},Felt,CushionX,side);
        }
        s.add({qx-.032f,0,-.581f},{qx+.032f,.044f,.581f},Felt,CushionZ,side);
        for(int k=-1;k<=1;++k) {
            float cx=k*pocket_x,cz=side*pocket_z;
            s.add({cx-bore,-.225f,cz-bore},{cx+bore,pocket_top,cz+bore},Leather,Well);
        }
    }
    // Keep the transmissive reference object out of the primary table image.
#ifdef ACG_GLASS_REFERENCE
    s.add({-.852f,.079f,-.822f},{-.682f,.249f,-.652f},Glass,Globe,2);
#endif
    s.add({.635f,.118f,-.744f},{.957f,.266f,-.743f},Grille,Screen,2);
    s.build(0,s.size); return s;
}
static constexpr StaticScene host_room=make_room();
#if defined(__CUDACC__) || defined(__CUDA__)
static __device__ __constant__ const StaticScene device_room=make_room();
#endif
ACG_HD inline const StaticScene& room() {
#if defined(__CUDA_ARCH__)
    return device_room;
#else
    return host_room;
#endif
}

// Slot 0 is the cue ball. The other slots run down the rows of a touching
// equilateral rack. Each row fans into a separate lane; trajectories are
// scripted (not a physics simulation) and finish inside the cushion noses.
ACG_HD inline Vec resting(int slot,int &row,int &col) {
    row=0; col=slot-1;
    if(slot==0) return {-.99f,radius,-.007f};
    while(col>row) { col-=row+1; ++row; }
    return {.31f+1.732051f*radius*row,radius,radius*(2*col-row)};
}
ACG_HD inline float ease_out(float t) { float a=1-clamp(t,0,1); return 1-a*a; }
constexpr float contact_t=1.12f, cue_contact=.31f-2*radius;
ACG_HD inline Vec cue_at_contact() { return {cue_contact,radius,0}; }
struct BallBVH;
struct Balls { Vec c[16]; Box box; const BallBVH* tree=nullptr; };
ACG_HD inline void place_balls(Balls &balls,float t,const float *positions) {
    if(positions) {
        for(int i=0;i<16;++i) {
            Vec p={positions[2*i],radius,positions[2*i+1]};
            balls.c[i]=p;
            if(i==0) balls.box={p,p}; else balls.box=unite(balls.box,{p,p});
        }
        balls.box.a=balls.box.a-grey(radius+eps); balls.box.b=balls.box.b+grey(radius+eps);
        return;
    }
    float spread=ease_out((t-contact_t)/2.85f),late=ease_out((t-3.97f)/1.45f);
    for(int i=0;i<16;++i) {
        int row,col; Vec p=resting(i,row,col);
        if(i==0) {
            if(t<=contact_t) p=mix(p,cue_at_contact(),clamp((t-.50f)/(contact_t-.50f),0,1));
            else p=mix(cue_at_contact(),{-1.03f,radius,.34f},spread);
        } else if(t>contact_t) {
            float rank=float(2*col-row);
            Vec end={-.78f+.435f*row+.044f*rank,radius,
                     .124f*rank+.050f*sinf(2.41f*i)};
            p=mix(p,end,spread);
            if(i==1 || i==12 || i==7) { p.x+=late*(i==1?-.108f:.093f); p.z+=late*.032f; }
        }
        balls.c[i]=p;
        if(i==0) balls.box={p,p}; else balls.box=unite(balls.box,{p,p});
    }
    balls.box.a=balls.box.a-grey(radius+eps); balls.box.b=balls.box.b+grey(radius+eps);
}
struct Hit { float t; Vec normal; Surface mat; int id; };
ACG_HD inline void record(Hit &h,float t,Vec n,Surface mat,int id=-1) { h={t,n,mat,id}; }
ACG_HD inline bool sphere_t(Ray r,Vec c,float rad,float tmin,float tmax,float &t) {
    Vec oc=c-r.o; float proj=dot(oc,r.d);
    // Perpendicular-distance form avoids subtracting room-scale squared
    // distances when resolving millimetre-wide silhouettes.
    Vec perp=oc-r.d*proj; float q=rad*rad-dot(perp,perp);
    if(q<0) return false;
    float root=sqrtf(q); t=proj-root; if(t<=tmin) t=proj+root;
    return t>tmin && t<tmax;
}
#ifdef ACG_COUNT_TRAVERSAL
#if defined(__CUDACC__) || defined(_OPENMP)
#error ACG_COUNT_TRAVERSAL requires a serial CPU build
#endif
struct BallCounters { uint64_t rays=0, boxes=0, spheres=0; };
inline BallCounters ball_counters;
#endif
#include "ball_bvh.h"
ACG_HD inline float pocket_distance2(Vec v) {
    float x=fabsf(v.x); x=x>pocket_x*.5f?x-pocket_x:x;
    float z=fabsf(v.z)-pocket_z; return x*x+z*z;
}
ACG_HD constexpr Vec basis_axis(int k,float s) {
    return k==0?Vec{s,0,0}:k==1?Vec{0,s,0}:Vec{0,0,s};
}
struct Span { float a,b; Vec na,nb; };
ACG_HD inline bool box_span(Box b,Ray r,Vec inv,Span &s) {
    Vec p=(b.a-r.o)*inv,q=(b.b-r.o)*inv;
    Vec l={lo(p.x,q.x),lo(p.y,q.y),lo(p.z,q.z)},u={hi(p.x,q.x),hi(p.y,q.y),hi(p.z,q.z)};
    int a=l.x>l.y?0:1; if(l.z>axis(l,a)) a=2;
    int e=u.x<u.y?0:1; if(u.z<axis(u,e)) e=2;
    s.a=axis(l,a); s.b=axis(u,e);
    if(s.b<s.a) return false;
    s.na=basis_axis(a,axis(inv,a)>0?-1.f:1.f); s.nb=basis_axis(e,axis(inv,e)>0?1.f:-1.f);
    return true;
}
// Clip the ray interval to dot(n,p-anchor) <= offset (n need not be unit).
ACG_HD inline bool clip(Span &s,Ray r,Vec n,Vec anchor,float offset) {
    float den=dot(n,r.d),num=offset-dot(n,r.o-anchor);
    if(fabsf(den)<1.e-12f) return num>=0;
    float t=num/den;
    if(den<0 && t>s.a) { s.a=t; s.na=n; }
    if(den>0 && t<s.b) { s.b=t; s.nb=n; }
    return s.a<=s.b;
}
ACG_HD inline void primitive(const Prim &p,Ray r,Vec inv,float t0,Hit &h) {
    Box b=p.box;
    if(p.shape==Globe) {
        Vec c=(b.a+b.b)*.5f; float t;
        if(sphere_t(r,c,(b.b.x-b.a.x)*.5f,t0,h.t,t)) record(h,t,unit((r.o-c)+r.d*t),Glass);
        return;
    }
    if(p.shape==Screen) {
        if(fabsf(r.d.z)<1.e-12f) return;
        float t=(b.b.z-r.o.z)*inv.z;
        if(t>t0 && t<h.t) {
            Vec v=r.o+r.d*t;
            if(v.x>=b.a.x && v.x<=b.b.x && v.y>=b.a.y && v.y<=b.b.y) record(h,t,{0,0,1},Grille);
        }
        return;
    }
    Span s;
    if(!box_span(b,r,inv,s) || s.b<=t0 || s.a>=h.t) return;
    if(p.shape==Well) {
        const Vec c=(b.a+b.b)*.5f,v=r.o-c;
        const float seam=pocket_top-pocket_bevel;
        const float bottom=(b.a.y-r.o.y)*inv.y;
        if(bottom>t0 && bottom<h.t) {
            Vec p=v+r.d*bottom;
            if(p.x*p.x+p.z*p.z<=pocket_throat*pocket_throat)
                record(h,bottom,{0,1,0},Leather);
        }
        // Intersect a vertical throat and a flared entrance, both analytically.
        for(int part=0;part<2;++part) {
            float k=part?pocket_slope:0.f;
            float r0=pocket_throat+k*(r.o.y-seam), rd=k*r.d.y;
            float a=r.d.x*r.d.x+r.d.z*r.d.z-rd*rd;
            float beta=v.x*r.d.x+v.z*r.d.z-r0*rd;
            float cc=v.x*v.x+v.z*v.z-r0*r0;
            float det=beta*beta-a*cc;
            if(det<0) continue;
            const float q=-beta-copysignf(sqrtf(det),beta);
            const bool linear=fabsf(a)<1.e-9f;
            for(int j=0;j<2;++j) {
                if((linear && (j || fabsf(beta)<1.e-9f)) || (j && q==0)) continue;
                float t=linear?-cc/(2*beta):(j?cc/q:q/a);
                if(t<=t0 || t>=h.t) continue;
                Vec p=v+r.d*t; float y=r.o.y+r.d.y*t;
                if(y>=(part?seam:b.a.y) && y<=(part?pocket_top:seam))
                    record(h,t,unit({-p.x,k*(pocket_throat+k*(y-seam)),-p.z}),
                           part?Ebony:Leather);
            }
        }
        return;
    }
    bool cushion=p.shape==CushionX || p.shape==CushionZ;
    if(cushion) {
        Vec along=p.shape==CushionX?Vec{1,0,0}:Vec{0,0,1};
        Vec out=p.shape==CushionX?Vec{0,0,float(p.side)}:Vec{float(p.side),0,0};
        Vec anchor=(b.a+b.b)*.5f-out*.032f; anchor.y=.029f;
        float length=.5f*dot(b.b-b.a,along);
        if(!clip(s,r,-out+Vec{0,.83f,0},anchor,0) || !clip(s,r,-out+Vec{0,-.46f,0},anchor,0) ||
           !clip(s,r,along-out*.58f,anchor,length-.025f) || !clip(s,r,-along-out*.58f,anchor,length-.025f)) return;
    }
    for(int k=0;k<2;++k) {
        float t=k?s.b:s.a; if(t<=t0 || t>=h.t) continue;
        if(p.shape==BoredBlock || cushion) {
            const Vec p0=r.o+r.d*t;
            const float opening=pocket_throat+pocket_slope*clamp(p0.y-(pocket_top-pocket_bevel),0.f,pocket_bevel);
            if(pocket_distance2(p0)<opening*opening) continue;
        }
        Vec n=k?s.nb:s.na; record(h,t,cushion?unit(n):n,p.mat); break;
    }
}

constexpr float light_y=1.37f,light_z=-1.08f,light_hx=.79f,light_hz=.215f;
constexpr float light_area=4*light_hx*light_hz;
ACG_HD constexpr Vec emission() { return {12.0f,10.9f,9.4f}; }
#include "soft_intervals.h"
ACG_HD inline bool intersect(Ray r,const Balls &balls,int stage,float t0,float t1,Hit &h) {
    h.t=t1; Vec inv=invdir(r.d);
#ifdef ACG_COUNT_TRAVERSAL
    ++ball_counters.rays;
    if(!balls.tree) ++ball_counters.boxes;
#endif
    if(balls.tree) {
        intersect_ball_bvh(*balls.tree,balls,r,inv,t0,h);
    } else if(overlaps(balls.box,r,inv,t0,h.t)) {
        for(int i=0;i<16;++i) {
#ifdef ACG_COUNT_TRAVERSAL
            ++ball_counters.spheres;
#endif
            float t;
            if(sphere_t(r,balls.c[i],radius,t0,h.t,t)) record(h,t,unit((r.o-balls.c[i])+r.d*t),Ball,i);
        }
    }
    const StaticScene &s=room();
    for(int i=0;i<s.nodes;) {
        const Node &n=s.n[i];
        if(!overlaps(n.box,r,inv,t0,h.t)) { i=n.escape; continue; }
        for(int j=n.first;j<n.first+n.count;++j) {
            const Prim &p=s.p[j]; if(stage!=0 || p.side!=2) primitive(p,r,inv,t0,h);
        }
        ++i;
    }
    if(stage==2 && fabsf(r.d.y)>1.e-12f) {
        float t=(light_y-r.o.y)*inv.y;
        if(t>t0 && t<h.t) {
            Vec p=r.o+r.d*t;
            if(fabsf(p.x)<light_hx && fabsf(p.z-light_z)<light_hz) record(h,t,{0,-1,0},Lamp);
        }
    }
    return h.t<t1;
}

// Original openwork motif, evaluated identically on camera, bounce and shadow
// rays. A partially covered inner tracery illustrates non-binary alpha as well.
ACG_HD inline float coverage(Vec p) {
#ifdef ACG_OPAQUE_MASK
    return 1;
#endif
    float x=p.x-.635f,y=p.y-.118f;
    if(x<.008f || y<.008f || x>.314f || y>.140f) return 1;
    float u=fabsf(fract(x/.046f)-.5f),v=fabsf(fract(y/.046f)-.5f);
    if(u+v>.69f || fabsf(u+v-.52f)<.051f) return 1;
    return u*u+v*v<.019f?.65f:0.f;
}
ACG_HD inline int number_for(int i) {
    // Eight in the middle; stripes and solids on opposite rear corners.
    switch(i) { case 2:return 11; case 11:return 2; case 5:return 8; case 8:return 5;
                case 4:return 13; case 13:return 4; default:return i; }
}
ACG_HD inline Vec pigment(int n) {
    if(n==0) return {.93f,.858f,.707f}; if(n==8) return {.011f,.014f,.019f};
    switch((n-1)%8) {
        case 0:return {.94f,.56f,.034f}; case 1:return {.025f,.105f,.55f};
        case 2:return {.68f,.034f,.036f}; case 3:return {.231f,.055f,.37f};
        case 4:return {.89f,.196f,.021f}; case 5:return {.025f,.305f,.133f};
        default:return {.37f,.040f,.054f};
    }
}
ACG_HD inline float grain(Vec p) {
    float bend=2.8f*sinf(7.1f*p.x+2.9f*p.z)+1.5f*sinf(17.3f*p.x+7*p.y);
    float rings=sinf(96*p.z+59*p.y+bend);
    return clamp(.53f+.27f*rings+.12f*sinf(239*p.z+123*p.y+2.3f*bend)+.06f*sinf(611*p.z+2*bend),0,1);
}
// Original bitmap digits, three columns by five rows, top row in high bits.
ACG_HD inline unsigned glyph(int n) {
    switch(n) {
        case 0:return 075557; case 1:return 026227; case 2:return 071747;
        case 3:return 071717; case 4:return 055711; case 5:return 074717;
        case 6:return 074757; case 7:return 071122; case 8:return 075757; default:return 075717;
    }
}
ACG_HD inline bool digit(int n,float u,float v) {
    int x=int(floorf(u)),y=int(floorf(v));
    return x>=0 && x<3 && y>=0 && y<5 && ((glyph(n)>>(14-3*y-x))&1u);
}
ACG_HD inline Vec unroll(Vec v,Vec delta) {
    float dist=sqrtf(delta.x*delta.x+delta.z*delta.z);
    return dist<eps?v:rotate(v,{delta.z/dist,0,-delta.x/dist},-dist/radius);
}
ACG_HD inline Vec ball_local(Vec n,int slot,Vec center,float t) {
    int row,col; Vec start=resting(slot,row,col);
    if(slot==0 && t>contact_t) {
        n=unroll(n,center-cue_at_contact()); n=unroll(n,cue_at_contact()-start);
    } else {
        if(t>3.97f && (slot==1 || slot==7 || slot==12)) {
            Vec d=Vec{slot==1?-.108f:.093f,0,.032f}*ease_out((t-3.97f)/1.45f);
            n=unroll(n,d); center=center-d;
        }
        n=unroll(n,center-start);
    }
    n=rotate(n,{0,1,0},-.38f+.41f*sinf(3.7f*slot));
    return rotate(n,{1,0,0},.61f+.13f*sinf(1.91f*slot));
}
ACG_HD inline Vec enamel(int num,Vec n,Vec body) {
    if(num==0) {
        return hi(fabsf(n.x),hi(fabsf(n.y),fabsf(n.z)))>.994f?Vec{.48f,.046f,.041f}:body;
    }
    // Longitude/latitude charts give two antipodal number badges and the belt.
    float lon=atan2f(n.x,n.z),lat=asinf(clamp(n.y,-1,1));
    if(lon>pi*.5f) lon-=pi; if(lon<-pi*.5f) lon+=pi;
    Vec ivory={.90f,.835f,.686f};
    Vec c=num>8 && fabsf(lat)>.49f?ivory:body;
    float rr=lon*lon+lat*lat;
    if(rr<.45f*.45f) {
        c=rr>.412f*.412f?mix(body,grey(.025f),.48f):ivory;
        float u=lon/.092f,v=2.5f-lat/.095f;
        bool ink=num<10?digit(num,u+1.5f,v):(digit(1,u+3.35f,v)||digit(num%10,u-.35f,v));
        if(ink) c={.013f,.017f,.022f};
    }
    return c;
}

struct Material { Vec diffuse,f0; float rough; };
ACG_HD inline Material varnish(Vec c,float rough) { return {c,grey(.044f),rough}; }
ACG_HD inline Material material(Hit h,Vec p,const Balls &balls,float t,int stage) {
    if(stage==0) {
        Vec c;
        switch(h.mat) {
            case Felt:c={.035f,.30f,.175f};break; case Ball:c=pigment(number_for(h.id));break;
            case Brass:c={.55f,.33f,.109f};break; case Timber:c={.245f,.091f,.036f};break;
            case Floor:c={.17f,.136f,.106f};break; case Ceiling:case Wall:c=grey(.48f);break;
            default:c=grey(.035f);break;
        }
        return {c,grey(0),1};
    }
    switch(h.mat) {
        case Ball: {
            int num=number_for(h.id); Vec n=ball_local(h.normal,h.id,balls.c[h.id],t);
            if(num==2) { // No diffuse lobe: tinted polished copper.
                return {grey(0),{.954f,.612f,.359f},.145f};
            }
            Vec base=pigment(num); float rough=.15f;
            if(num==5) { // Marquetry, with a clear dielectric finish (not a metal).
                base=mix({.112f,.031f,.012f},{.64f,.301f,.088f},grain(n*.53f)); rough=.205f;
            }
            if(num==4) { // Veined marble, evaluated in ball-local coordinates.
                float bend=.21f*sinf(8*n.z)+.09f*sinf(19*n.x+11*n.z);
                float vein=1-fabsf(sinf(13*(n.y+bend)));
                vein*=vein; vein*=vein; vein*=vein;
                base=mix({.76f,.73f,.66f},{.14f,.18f,.23f},vein); rough=.23f;
            } else if(num==6) { // Layered ceramic glaze with mottled pigment.
                float mottling=.5f+.5f*sinf(17*n.x+3*sinf(13*n.z))*sinf(19*n.y);
                base=mix({.016f,.085f,.19f},{.16f,.58f,.66f},mottling); rough=.18f;
            } else if(num==7) { // Basket weave with alternating over-under threads.
                float a=atan2f(n.x,n.z)*(12/pi), b=asinf(clamp(n.y,-1,1))*(12/pi);
                float warp=1-fabsf(2*fract(a)-1), weft=1-fabsf(2*fract(b)-1);
                bool over=fract((floorf(a)+floorf(b))*.5f)<.25f;
                base=over?mix({.083f,.029f,.017f},{.58f,.28f,.15f},warp)
                         :mix({.064f,.037f,.021f},{.38f,.24f,.11f},weft);
                rough=.46f;
            }
            return varnish(enamel(num,n,base),rough);
        }
        case Felt: {
            float weave=.965f+.023f*sinf(2480*p.x+61*p.z)*sinf(2175*p.z);
            return {{.025f*weave,.256f*weave,.16f*weave},grey(.019f),.81f};
        }
        case Timber: {
            bool end=fabsf(p.x)>1.32f && fabsf(p.z)<.69f;
            Vec wood=end?Vec{p.z,p.y,p.x}:p;
            Vec c=mix({.070f,.021f,.012f},{.332f,.13f,.048f},grain(wood));
            if(h.normal.y>.95f && fabsf(p.y-.048f)<.001f) {
                float a=end?p.z:p.x, b=end?fabsf(p.x)-1.376f:fabsf(p.z)-.742f;
                float d=fabsf(fract(a/.3175f+.5f)-.5f)*.3175f;
                if(d/.012f+fabsf(b)/.018f<1) c={.89f,.82f,.63f};
                if(pocket_distance2(p)<.079f*.079f) c={.049f,.031f,.022f};
            }
            return varnish(c,.255f);
        }
        case Brass:case Grille:return {grey(0),{.83f,.592f,.283f},h.mat==Grille?.31f:.235f};
        case Ebony:return varnish({.042f,.026f,.023f},.31f);
        case Leather:return varnish({.022f,.015f,.012f},.73f);
        case Wall: {
            float z=fabsf(h.normal.x)>.5f?p.z:p.x;
            bool molding=fabsf(p.y-.12f)<.022f || (p.y<.12f && fract(z/.46f)<.048f);
            Vec c=p.y>.142f?Vec{.32f,.364f,.351f}:(molding?Vec{.081f,.105f,.094f}:Vec{.124f,.151f,.137f});
            return varnish(c,.74f);
        }
        case Floor: {
            float row=floorf(p.z/.155f),u=fract(p.x/1.07f+row*.38197f),v=fract(p.z/.155f);
            Vec c=mix({.086f,.051f,.030f},{.291f,.177f,.095f},grain({p.x,p.y+row,p.z}));
            return varnish(c*((v<.008f || u<.0014f)?.48f:.93f+.07f*sinf(row+floorf(p.x/1.07f+row*.38197f))),.43f);
        }
        default:return varnish({.67f,.65f,.594f},.82f);
    }
}

struct Frame { Vec x,y,z; };
ACG_HD inline Frame frame(Vec z) {
    Vec x=unit(cross(fabsf(z.y)<.92f?Vec{0,1,0}:Vec{0,0,1},z)); return {x,cross(z,x),z};
}
ACG_HD inline Vec world(Frame f,Vec a) { return f.x*a.x+f.y*a.y+f.z*a.z; }
ACG_HD inline Vec local(Frame f,Vec a) { return {dot(a,f.x),dot(a,f.y),dot(a,f.z)}; }
ACG_HD inline Vec fresnel(Vec f0,float c) {
    float x=clamp(1-c,0,1),x2=x*x; return f0+(grey(1)-f0)*(x2*x2*x);
}
ACG_HD inline float spec_prob(Material m) {
    float d=peak(m.diffuse),s=peak(m.f0); if(s==0) return 0; if(d==0) return 1;
    return clamp(s/(s+d),.16f,.8f);
}
ACG_HD inline float smith_g1(float c,float a2) { return 2*c/(c+sqrtf(a2+(1-a2)*c*c)); }
// BRDF and total mixture PDF in solid angle. The GGX sampler below samples
// visible half-normals, so reflection PDF = D(h) G1(wo) / (4 n.wo).
ACG_HD inline Vec brdf(Material m,Vec n,Vec wo,Vec wi,float &pdf) {
    float a=dot(n,wo),b=dot(n,wi); pdf=0; if(a<=0 || b<=0) return grey(0);
    float ps=spec_prob(m); pdf=(1-ps)*b/pi;
    if(ps==0) return m.diffuse*(1/pi);
    Vec h=unit(wo+wi); float nh=clamp(dot(n,h),0,1);
    float alpha=m.rough*m.rough,a2=alpha*alpha;
    float d=(1-nh*nh)+nh*nh*a2,D=a2/(pi*d*d),g=smith_g1(a,a2);
    pdf+=ps*D*g/(4*a);
    Vec F=fresnel(m.f0,dot(wi,h));
    return (grey(1)-F)*m.diffuse*(1/pi)+F*(D*g*smith_g1(b,a2)/(4*a*b));
}
ACG_HD inline Vec sample_direction(Material m,Vec n,Vec wo,Rng &rng) {
    Frame f=frame(n); float choice=rng.next(),u=rng.next(),v=rng.next();
    float rad=sqrtf(u),ang=2*pi*v,dx=rad*cosf(ang),dy=rad*sinf(ang);
    if(choice>=spec_prob(m)) return world(f,{dx,dy,sqrtf(1-u)});
    float a=m.rough*m.rough; Vec view=local(f,wo),vh=unit({a*view.x,a*view.y,view.z});
    float l=vh.x*vh.x+vh.y*vh.y;
    Vec t1=l>1.e-15f?Vec{-vh.y,vh.x,0}*(1/sqrtf(l)):Vec{1,0,0};
    float s=.5f*(1+vh.z); dy=(1-s)*sqrtf(hi(0,1-dx*dx))+s*dy;
    Vec h=t1*dx+cross(vh,t1)*dy+vh*sqrtf(hi(0,1-dx*dx-dy*dy));
    Vec wh=world(f,unit({a*h.x,a*h.y,hi(0,h.z)})); return unit(wh*(2*dot(wo,wh))-wo);
}
ACG_HD inline float power_weight(float a,float b) { return a*a/(a*a+b*b); }
ACG_HD inline float visibility(Ray r,float limit,const Balls &balls,int stage) {
    Hit h; float through=1,t0=eps;
    while(intersect(r,balls,stage,t0,limit,h)) {
        if(h.mat!=Grille) return 0; // A thick refractor is not a straight shadow path.
        through*=1-coverage(r.o+r.d*h.t); if(through==0) break;
        t0=h.t+eps;
    }
    return through;
}
constexpr int area_samples=2;
ACG_HD inline Vec direct_sample(Vec p,Vec n,Vec wo,Vec source,Vec intensity,bool area,
                                Material m,const Balls &balls,int stage,float cached=-1) {
    Vec delta=source-p; float d2=dot(delta,delta),dist=sqrtf(d2); Vec wi=delta*(1/dist);
    float c=dot(n,wi); if(c<=0 || (area && wi.y<=0)) return grey(0);
    // Aim from the offset origin to the actual sample (including near grazing).
    Vec org=p+n*eps,sd=source-org; float sl=sqrtf(dot(sd,sd));
    float vis=cached>=0?cached:visibility({org,sd*(1/sl)},sl-3*eps,balls,stage); if(vis==0) return grey(0);
    float pdf; Vec f=brdf(m,n,wo,wi,pdf);
    if(!area) return f*intensity*(vis*c/d2); // radiant intensity, inverse square
    float lp=d2/(wi.y*light_area);
    return f*intensity*(vis*c*power_weight(area_samples*lp,pdf)/(lp*area_samples));
}
#ifdef __CUDACC__
ACG_HD __noinline__
#else
inline
#endif
bool analytic_area_sample(Vec p,Vec n,Vec wo,Vec q,Material m,const Balls& balls,Vec& result) {
    SoftSlice slice;
    if(!soft_slice(p,balls.c,q.z,slice,soft_lamp,radius,eps)) return false;
    Vec c=m.diffuse*(1/pi);
    Vec sum=c*float(slice.geometry);
    if(soft_visible(slice,q.x)) {
        Vec d=q-p; float d2=dot(d,d); Vec wi=d*(1/sqrtf(d2));
        float pdf; Vec f=brdf(m,n,wo,wi,pdf);
        float lp=d2/(wi.y*light_area);
        sum=sum+(f*power_weight(area_samples*lp,pdf)-c)*(2*light_hx*wi.y*wi.y/d2);
    }
    result=emission()*sum*(2*light_hz/area_samples);
    return true;
}
ACG_HD inline Vec lighting(Vec p,Vec n,Vec wo,Material m,const Balls &balls,int stage,Rng &rng,const Hit& hit,bool analytic,float cached) {
    Vec I=stage==2?Vec{1.8f,1.63f,1.52f}:Vec{8.05f,7.35f,6.37f};
    Vec out=direct_sample(p,n,wo,{-.73f,1.54f,-.46f},I,false,m,balls,stage,cached);
    if(stage==2 && p.y<light_y) {
        const bool eligible=analytic && soft_eligible(stage,hit,p,n,balls.c);
#ifdef ACG_STRATIFY_LIGHT_2D
        // Randomly pair the X and Z strata; a fixed diagonal would leave half
        // of the emitter unsampled. The marginal light PDF is unchanged.
        static_assert(area_samples==2,"two strata per axis");
        const int flip=int(hash32(rng.s ^ 0x640f32d1u)>>31);
#endif
        for(int i=0;i<area_samples;++i) {
            float u=(i+rng.next())/area_samples,v=rng.next();
#ifdef ACG_STRATIFY_LIGHT_2D
            if(!eligible) v=((i^flip)+v)*.5f;
#endif
            Vec q={(2*(eligible?u*area_samples-i:u)-1)*light_hx,light_y,light_z+(2*v-1)*light_hz};
            Vec contribution;
            if(eligible && analytic_area_sample(p,n,wo,q,m,balls,contribution)) {
                out=out+contribution;
                continue;
            }
            out=out+direct_sample(p,n,wo,q,emission(),true,m,balls,stage);
        }
    }
    return out;
}
ACG_HD inline bool glass_bounce(Vec d,Vec n,bool entering,Rng &rng,Vec &dir,float &weight) {
    float eta=entering?1/1.52f:1.52f,ci=clamp(-dot(d,n),0,1);
    float ct2=1-eta*eta*(1-ci*ci),ct=sqrtf(hi(0,ct2)),R=1;
    if(ct2>0) {
        float rs=(eta*ci-ct)/(eta*ci+ct),rp=(ci-eta*ct)/(ci+eta*ct); R=.5f*(rs*rs+rp*rp);
    }
    if(rng.next()<R) { dir=unit(d+n*(2*ci)); weight=1; return false; }
    dir=unit(d*eta+n*(eta*ci-ct)); weight=eta*eta; return true;
}
ACG_HD inline Vec trace(Ray ray,const Balls &balls,int stage,float t,Rng &rng,bool analytic=false,float cached=-1) {
    Vec out={},beta=grey(1),prev={}; float prev_pdf=0,eta_scale=1; bool delta=true,inside=false;
    for(int depth=0;depth<14;++depth) {
        Hit hit; float t0=eps;
        for(;;) {
            if(!intersect(ray,balls,stage,t0,far_t,hit)) return out;
            if(hit.mat!=Grille || rng.next()<coverage(ray.o+ray.d*hit.t)) break;
            t0=hit.t+eps;
        }
        Vec p=ray.o+ray.d*hit.t,wo=-ray.d;
        bool front=dot(wo,hit.normal)>0; Vec n=front?hit.normal:-hit.normal;
        if(inside) beta=beta*Vec{expf(-1.63f*hit.t),expf(-.24f*hit.t),expf(-.45f*hit.t)};
        if(hit.mat==Lamp) {
            if(front) {
                float w=1;
                if(!delta) {
                    Vec d=p-prev; float d2=dot(d,d),lp=d2*sqrtf(d2)/(light_area*d.y);
                    w=power_weight(prev_pdf,area_samples*lp);
                }
                out=out+beta*emission()*w;
            }
            break;
        }
        Vec wi;
        if(hit.mat==Glass) {
#ifdef ACG_NO_REFRACTION
            wi=ray.d-n*(2*dot(ray.d,n));
#else
            float w; bool transmit=glass_bounce(ray.d,n,front,rng,wi,w);
            if(transmit) { inside=front; beta=beta*w; eta_scale/=w; }
#endif
            delta=true;
        } else {
            Material m=material(hit,p,balls,t,stage);
            const float vis=depth==0 && hit.mat==Felt && n.y==1 && fabsf(p.y)<eps?cached:-1.f;
            out=out+beta*lighting(p,n,wo,m,balls,stage,rng,hit,analytic,vis);
            if(stage==0) break;
            wi=sample_direction(m,n,wo,rng); float c=dot(wi,n);
            if(c<=0) break;
            Vec f=brdf(m,n,wo,wi,prev_pdf); if(prev_pdf<=0) break;
            beta=beta*f*(c/prev_pdf); prev=p; delta=false;
        }
        if(depth>=3) {
            float q=clamp(peak(beta)*eta_scale,.08f,.95f); if(rng.next()>=q) break; beta=beta*(1/q);
        }
        ray={p+n*(dot(wi,n)>0?eps:-eps),wi};
    }
    return out;
}
ACG_HD inline Ray camera(int x,int y,int w,int h,int stage,Rng &rng) {
    constexpr float camera_height=light_y-.12f; // Below the overhead emitter.
    const float aspect=float(w)/float(h),halfheight=hi(.77f,1.40f/aspect);
    float jx=rng.next(), jy=rng.next();
#ifdef ACG_NO_AA
    jx=jy=.5f;
#endif
    float u=(2*(float(x)+jx)/w-1)*aspect*halfheight;
    float v=(1-2*(float(y)+jy)/h)*halfheight;
    const Vec origin={u,camera_height,-v};
    const Vec down={0,-1,0};
    if(stage==0) return {origin,down};
    // Orthographic thin lens: zero aperture gives parallel primary rays.
    float rad=.004f*sqrtf(rng.next()),phi=2*pi*rng.next();
    Vec lens={rad*cosf(phi),0,rad*sinf(phi)};
#ifdef ACG_PINHOLE
    lens={0,0,0};
#endif
    return {origin+lens,unit(down*(camera_height-radius)-lens)};
}
} // namespace acg_scene_detail

// One independent, untonemapped radiance sample; pixel (0,0) is top-left.
// Change seed for each sample, then average in linear RGB before display.
// balls: optional 16 packed (x,z) centre pairs in metres about table centre.
// A supplied physics snapshot is frozen for the entire path; for motion blur
// the caller supplies snapshots at shutter times. Without one, time is seconds
// on the built-in demonstration and each path samples a centred 1/48 s shutter.
ACG_HD inline Vec sample_pixel(int x,int y,int w,int h,int stage,unsigned seed,float time,
                               const float *balls=nullptr,const acg_scene_detail::BallBVH* tree=nullptr,bool analytic=false,float cached=-1) {
    using namespace acg_scene_detail;
    Rng rng={hash32(seed^hash32(uint32_t(x)+0x9e3779b9u)^hash32(uint32_t(y)+0x243f6a88u))};
    stage=stage<0?0:(stage>2?2:stage);
    float t=balls?0.f:(stage==0?time:time+(rng.next()-.5f)*(1.f/48));
    Balls poses; place_balls(poses,t,balls);
    poses.tree=tree;
    Ray r=camera(x,y,w,h,stage,rng); return trace(r,poses,stage,t,rng,analytic,cached);
}
#endif
