#ifndef ACG_SCENE_H_INCLUDED
#define ACG_SCENE_H_INCLUDED

#include <math.h>
#include <stdint.h>
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__CUDA__)
#include <arm_neon.h>
#include <array>
#include <cstring>
#include <limits>
#define ACG_PACKED_ROOM 1
#endif

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

enum Surface { Felt, Timber, Brass, Ebony, Leather, Floor, Wall, Ceiling, Glass, Grille, Ball, Lamp, CushionCloth };
enum Shape { Block, BoredBlock, CushionX, CushionZ, Well, Globe, Screen, RailX, RailZ };
ACG_HD constexpr bool uses_box_span(Shape shape) {
    return shape!=Globe && shape!=Screen;
}
struct Prim { Box box; Shape shape; Surface mat; int side; };
constexpr int static_scene_capacity=96;
#ifdef ACG_PACKED_ROOM
static_assert(static_scene_capacity<=UINT8_MAX,"Static BVH indices must fit in one byte");
struct Node {
    Box box{};
    uint8_t first=0,count=0,escape=0;
    ACG_HD constexpr Node()=default;
    ACG_HD constexpr Node(Box box,int first,int count,int escape)
        :box(box),first(static_cast<uint8_t>(first)),count(static_cast<uint8_t>(count)),
         escape(static_cast<uint8_t>(escape)) {}
};
#else
struct Node { Box box; int first,count,escape; };
#endif
// A small, immutable preorder BVH. Construction/sorting happen at compile time,
// not once per thread or per frame. Escape indices make traversal stackless.
struct StaticScene {
    Prim p[static_scene_capacity]; Node n[static_scene_capacity]; int size,nodes;
    ACG_HD constexpr void add(Vec a,Vec b,Surface m,Shape s=Block,int side=0) {
        p[size++]={{a,b},s,m,side};
    }
    template<int LeafSize>
    ACG_HD constexpr void build(int first,int count) {
        static_assert(LeafSize>=1 && LeafSize<=3,"Invalid static BVH leaf size");
        int k=nodes++; Box box=p[first].box;
        for(int i=1;i<count;++i) box=unite(box,p[first+i].box);
        n[k]={box,first,count,k+1};
        if(count<=LeafSize) return;
        if(count<=3) {
            // Refine ordinary leaves in the existing order to preserve hit ties.
            // Shapes that bypass the primitive slab test need the original leaf bounds.
            for(int i=first;i<first+count;++i)
                if(!uses_box_span(p[i].shape)) return;
        } else {
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
        }
        n[k].count=0;
        build<LeafSize>(first,count/2);
        build<LeafSize>(first+count/2,count-count/2);
        n[k].escape=nodes;
    }
};
constexpr float table_nose_x=1.270f,table_nose_z=.635f;
// 9 ft Gold Crown envelope: the gap between the two nose lines is unchanged.
constexpr float rail_top=.04445f,rail_width=.186f,cushion_width=.052f,featherstrip=.050f;
constexpr float rail_half_width=(rail_width-featherstrip)*.5f;
constexpr float rail_x=table_nose_x+featherstrip+rail_half_width;
constexpr float rail_z=table_nose_z+featherstrip+rail_half_width;
constexpr float pocket_x=1.287f,pocket_z=.652f,side_pocket_z=.686f,bore=.076f,pocket_tilt=.32f;
constexpr float pocket_bound=.100f; // Covers the two corner jaw tips as well as the rear curve.
constexpr float pocket_mouth_y=rail_top,pocket_throat=.056f,pocket_bevel=.025f;
constexpr float pocket_slope=(bore-pocket_throat)/pocket_bevel;
constexpr float pocket_seam=pocket_mouth_y-pocket_bevel;
// The visible liner narrows in two sections below the slate, rather than
// putting a vertical cylinder behind the mouth's small top bevel.
constexpr float pocket_floor_y=-.155f,pocket_floor_radius=.040f;
constexpr float pocket_neck_y=-.030f,pocket_neck_radius=.047f;
constexpr float pocket_lower_slope=(pocket_neck_radius-pocket_floor_radius)/(pocket_neck_y-pocket_floor_y);
constexpr float pocket_upper_slope=(pocket_throat-pocket_neck_radius)/(pocket_seam-pocket_neck_y);
constexpr float pocket_top=rail_top;
constexpr float cushion_nose_y=.635f*(2*radius),cushion_nose_round=.004f;
constexpr float cushion_shoulder=.0025f;
constexpr float cushion_rise=(rail_top-cushion_nose_y-cushion_shoulder)/featherstrip;
constexpr float side_mouth_half=.064f,corner_jaw_setback=.082f;
constexpr float cushion_bound_span=.642f;
constexpr float table_clear_x=1.20f,table_clear_z=.55f;
template<int LeafSize=3>
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
        const float z=side*rail_z,x=side*rail_x;
        constexpr float extent_x=rail_x+rail_half_width,extent_z=rail_z+rail_half_width;
        s.add({-extent_x,-.115f,z-rail_half_width},{extent_x,rail_top,z+rail_half_width},Timber,RailX,side);
        s.add({x-rail_half_width,-.115f,-extent_z},{x+rail_half_width,rail_top,extent_z},Timber,RailZ,side);
        const float qz=side*table_nose_z,qx=side*table_nose_x;
        const float az=qz+side*cushion_width,ax=qx+side*cushion_width;
        for(int half=-1;half<=1;half+=2) {
            float cx=half*.638f;
            s.add({cx-cushion_bound_span,0,lo(qz,az)},
                  {cx+cushion_bound_span,rail_top,hi(qz,az)},CushionCloth,CushionX,side);
        }
        s.add({lo(qx,ax),0,-cushion_bound_span},{hi(qx,ax),rail_top,cushion_bound_span},CushionCloth,CushionZ,side);
        for(int k=-1;k<=1;++k) {
            float cx=k*pocket_x,cz=side*(k?pocket_z:side_pocket_z);
            s.add({cx-pocket_bound,pocket_floor_y,cz-pocket_bound},
                  {cx+pocket_bound,pocket_top,cz+pocket_bound},Leather,Well);
        }
    }
    // Keep the transmissive reference object out of the primary table image.
#ifdef ACG_GLASS_REFERENCE
    s.add({-.852f,.079f,-.822f},{-.682f,.249f,-.652f},Glass,Globe,2);
#endif
    s.add({.635f,.118f,-.744f},{.957f,.266f,-.743f},Grille,Screen,2);
    s.build<LeafSize>(0,s.size); return s;
}
static constexpr StaticScene host_room=make_room<1>();

// Both room BVHs are built from the same primitive definitions.
static constexpr bool room_has_glass=[] {
    for(int i=0;i<host_room.size;++i) {
        const Prim& p=host_room.p[i];
        if(p.shape==Globe || p.mat==Glass) return true;
    }
    return false;
}();

#if !defined(__CUDA_ARCH__)
constexpr Box room_interior={{-3.6f,-.82f,-2.65f},{3.6f,2.55f,3.65f}};

constexpr bool outside_room(Box box) {
    return box.b.x<=room_interior.a.x || box.a.x>=room_interior.b.x ||
           box.b.y<=room_interior.a.y || box.a.y>=room_interior.b.y ||
           box.b.z<=room_interior.a.z || box.a.z>=room_interior.b.z;
}

// Group surviving leaves by surface area without changing their bounds or order.
struct OrderedTree {
    Node n[sizeof(host_room.n)/sizeof(Node)]{};
    int size=0;

    static constexpr float half_area(Box box) {
        const Vec extent=box.b-box.a;
        return extent.x*extent.y+extent.x*extent.z+extent.y*extent.z;
    }

    constexpr void build(const Node* leaves,int count) {
        const int entry=size++;
        if(count==1) {
            n[entry]=leaves[0];
            n[entry].escape=size;
            return;
        }
        Box suffix[host_room.size]{};
        suffix[count-1]=leaves[count-1].box;
        for(int i=count-2;i>=0;--i)
            suffix[i]=unite(leaves[i].box,suffix[i+1]);
        Box prefix=leaves[0].box;
        int split=1;
        float best=half_area(prefix)+(count-1)*half_area(suffix[1]);
        for(int i=2;i<count;++i) {
            prefix=unite(prefix,leaves[i-1].box);
            const float cost=i*half_area(prefix)+(count-i)*half_area(suffix[i]);
            if(cost<best) { best=cost; split=i; }
        }
        build(leaves,split);
        build(leaves+split,count-split);
        n[entry]={suffix[0],0,0,size};
    }
};

constexpr OrderedTree make_interior_tree() {
    Node leaves[host_room.size]{};
    int count=0;
    for(int i=0;i<host_room.nodes;++i) {
        const Node& leaf=host_room.n[i];
        for(int j=leaf.first;j<leaf.first+leaf.count;++j) {
            const Prim& prim=host_room.p[j];
            if(prim.shape!=Block || !outside_room(prim.box)) {
                leaves[count++]=leaf;
                break;
            }
        }
    }
    OrderedTree tree{};
    if(count) tree.build(leaves,count);
    return tree;
}

static constexpr OrderedTree interior_tree=make_interior_tree();

#ifdef ACG_PACKED_ROOM
constexpr int packed_index_bits=7;
constexpr uint32_t packed_index_mask=(1u<<packed_index_bits)-1;
static_assert(static_scene_capacity<=packed_index_mask,"Packed BVH indices must fit in seven bits");
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float)==sizeof(uint32_t),
              "Packed BVH metadata requires IEEE binary32");
struct alignas(32) PackedNode {
    float lower[4],upper[4];
};
static_assert(sizeof(PackedNode)==32);

template<int Count>
constexpr std::array<PackedNode,Count> pack_tree(const Node* source) {
    std::array<PackedNode,Count> tree{};
    for(int i=0;i<Count;++i) {
        const Node& node=source[i];
        const Box box=node.box;
        const uint32_t indices=uint32_t(node.first) | (uint32_t(node.count)<<packed_index_bits) |
                               (uint32_t(node.escape)<<(2*packed_index_bits));
        // Integers in [2^23, 2^24) have this payload in their binary32 fraction.
        // The packed slab test uses the spare lane for ray interval bounds.
        const float metadata=static_cast<float>((1u<<23)+indices);
        tree[i]={{box.a.x,box.a.y,box.a.z,metadata},{box.b.x,box.b.y,box.b.z,0.f}};
    }
    return tree;
}

// Input is the binary preorder from OrderedTree::build. Promote a child
// when its estimated acceptance rate makes testing its two children cheaper.
constexpr OrderedTree promote_children(const OrderedTree& source) {
    OrderedTree result{};
    const auto emit=[&](const auto& self,int index,float parent_area)->void {
        const Node& n=source.n[index];
        const float area=OrderedTree::half_area(n.box);
        const bool keep=index==0 || n.count || area<=.5f*parent_area;
        const int entry=result.size;
        if(keep) result.n[result.size++]=n;
        if(!n.count) {
            const int left=index+1;
            const float inherited=keep?area:parent_area;
            self(self,left,inherited);
            self(self,source.n[left].escape,inherited);
        }
        if(keep) result.n[entry].escape=result.size;
    };
    if(source.size) emit(emit,0,0);
    return result;
}

static constexpr OrderedTree wide_interior=promote_children(interior_tree);
static constexpr auto packed_interior=pack_tree<wide_interior.size>(wide_interior.n);
static constexpr OrderedTree grouped_room=[] {
    Node leaves[host_room.size]{};
    int count=0;
    for(int i=0;i<host_room.nodes;++i)
        if(host_room.n[i].count) leaves[count++]=host_room.n[i];
    OrderedTree tree{};
    if(count) tree.build(leaves,count);
    return tree;
}();
static constexpr OrderedTree wide_room=promote_children(grouped_room);
static constexpr auto packed_room=pack_tree<wide_room.size>(wide_room.n);
static constexpr auto above_rail_leaves=[] {
    struct LeafList {
        Node leaves[host_room.size]{};
        int count=0;
    } result;
    const auto excluded=[](int index) {
        const Prim& p=host_room.p[index];
        return uses_box_span(p.shape) &&
               (p.box.b.y<=pocket_top || (p.shape==Block && outside_room(p.box)));
    };
    for(int i=0;i<host_room.nodes;++i) {
        Node leaf=host_room.n[i];
        while(leaf.count && excluded(leaf.first)) { ++leaf.first; --leaf.count; }
        while(leaf.count && excluded(leaf.first+leaf.count-1)) --leaf.count;
        if(!leaf.count) continue;
        // Preserve the original gate for shapes that bypass the primitive slab test.
        leaf.escape=result.count+1;
        result.leaves[result.count++]=leaf;
    }
    return result;
}();
static constexpr auto packed_above_rails=
    pack_tree<above_rail_leaves.count>(above_rail_leaves.leaves);

// The scene bounds and the ray caps keep |(bound - origin) * inverse| below FLT_MAX.
constexpr Box packed_root=host_room.n[0].box;
static_assert(packed_root.a.x>=-0x1p59f && packed_root.a.y>=-0x1p59f && packed_root.a.z>=-0x1p59f &&
              packed_root.b.x<=0x1p59f && packed_root.b.y<=0x1p59f && packed_root.b.z<=0x1p59f,
              "Room bounds exceed the packed slab range");

inline bool packed_full_ray(Vec origin,Vec inverse) {
    return fabsf(origin.x)<=0x1p60f && fabsf(origin.y)<=0x1p60f && fabsf(origin.z)<=0x1p60f &&
           fabsf(inverse.x)<=0x1p67f && fabsf(inverse.y)<=0x1p67f && fabsf(inverse.z)<=0x1p67f;
}

struct PackedSlabs {
    float32x4_t origin,inverse;

    PackedSlabs(Vec ray_origin,Vec reciprocal,float t0)
        :origin{ray_origin.x,ray_origin.y,ray_origin.z,-t0},
         inverse{reciprocal.x,reciprocal.y,reciprocal.z,1.f} {}

    inline bool operator()(const PackedNode& node,float t1) const {
        const float32x4_t a=vmulq_f32(vsubq_f32(vld1q_f32(node.lower),origin),inverse);
        const float32x4_t b=vmulq_f32(vsubq_f32(vld1q_f32(node.upper),origin),inverse);
        // In lane four, b is t0 and a is t0 plus positive metadata.
        // The minimum is t0. The maximum is replaced with the live upper limit.
        const float32x4_t low=vminq_f32(a,b);
        const float32x4_t high=vsetq_lane_f32(t1,vmaxq_f32(a,b),3);
        return vmaxnmvq_f32(low)<=vminnmvq_f32(high);
    }
};
#endif

inline bool inside_slab(float origin,float inverse,float lo,float hi,float limit) {
    return origin>lo && origin<hi && limit<((inverse>0?hi:lo)-origin)*inverse;
}

inline bool inside_room(Vec origin,Vec inverse,float limit) {
    // Use the same signed reciprocal and slab arithmetic as intersection.
    return inside_slab(origin.x,inverse.x,room_interior.a.x,room_interior.b.x,limit) &&
           inside_slab(origin.y,inverse.y,room_interior.a.y,room_interior.b.y,limit) &&
           inside_slab(origin.z,inverse.z,room_interior.a.z,room_interior.b.z,limit);
}

#ifdef ACG_PACKED_ROOM
// Certify that every pruned primitive extending above the slate lies outside
// the central rectangle in XZ.
constexpr bool table_center_is_clear() {
    for(int i=0;i<host_room.size;++i) {
        const Prim& p=host_room.p[i];
        if(!uses_box_span(p.shape) || (p.shape==Block && outside_room(p.box))) continue;
        const Box b=p.box;
        if(b.b.y<=0 || b.b.y>pocket_top) continue;
        if(b.a.x<table_clear_x && b.b.x>-table_clear_x &&
           b.a.z<table_clear_z && b.b.z>-table_clear_z) return false;
    }
    return true;
}

inline bool clears_table_center(Vec origin,Vec inverse,float limit) {
    if constexpr(!table_center_is_clear()) return false;
    if(!(origin.y>=0)) return false;
    const float below_rails=lo(limit,(pocket_top-origin.y)*inverse.y);
    return inside_slab(origin.x,inverse.x,-table_clear_x,table_clear_x,below_rails) &&
           inside_slab(origin.z,inverse.z,-table_clear_z,table_clear_z,below_rails);
}
#endif
#endif
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
#ifdef ACG_PACKED_ROOM
struct alignas(16) PackedBalls { float x[16],z[16]; };
#endif
struct Balls {
    Vec c[16];
    Box box;
    const BallBVH* tree=nullptr;
#ifdef ACG_PACKED_ROOM
    const PackedBalls* packed=nullptr;
#endif
};
ACG_HD inline void place_balls(Balls &balls,float t,const float *positions) {
#ifdef ACG_PACKED_ROOM
    balls.packed=nullptr;
#endif
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

constexpr float max_overhead_lateral_speed=.01f;
constexpr float min_overhead_downward_speed=.99f;

#ifdef ACG_PACKED_ROOM
inline void prepare_packed_balls(Balls& balls,PackedBalls& packed) {
    for(int i=0;i<16;++i) {
        const Vec c=balls.c[i];
        if(!(fabsf(c.x)<=4 && c.y==radius && fabsf(c.z)<=4)) {
            balls.packed=nullptr;
            return;
        }
        packed.x[i]=c.x;
        packed.z[i]=c.z;
    }
    balls.packed=&packed;
}

inline bool packed_ball_ray(Ray r) {
    return fabsf(r.o.x)<=8 && fabsf(r.o.y)<=8 && fabsf(r.o.z)<=8 &&
           fabsf(r.d.x)<=1.001f && fabsf(r.d.y)<=1.001f && fabsf(r.d.z)<=1.001f;
}

inline uint32_t packed_ball_candidates(const PackedBalls& packed,Ray r) {
    const float32x4_t dy=vdupq_n_f32(radius-r.o.y);
    const float32x4_t ox=vdupq_n_f32(r.o.x), oz=vdupq_n_f32(r.o.z);
    // Each center-to-origin component is at most 12 m, and direction components
    // are at most 1.001. The 1 mm margin covers FMA round-off in the filter.
    // The original sphere test resolves hits and their distances.
    constexpr float padded=radius+.001f;
    const float32x4_t limit=vdupq_n_f32(padded*padded);
    const uint32x4_t lane_bits={1,2,4,8};
    uint32_t slots=0;
    for(int i=0;i<16;i+=4) {
        const float32x4_t dx=vsubq_f32(vld1q_f32(packed.x+i),ox);
        const float32x4_t dz=vsubq_f32(vld1q_f32(packed.z+i),oz);
        const float32x4_t proj=vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(dx,r.d.x),dy,r.d.y),dz,r.d.z);
        const float32x4_t px=vfmsq_n_f32(dx,proj,r.d.x);
        const float32x4_t py=vfmsq_n_f32(dy,proj,r.d.y);
        const float32x4_t pz=vfmsq_n_f32(dz,proj,r.d.z);
        const float32x4_t dist2=vfmaq_f32(vfmaq_f32(vmulq_f32(px,px),py,py),pz,pz);
        slots|=vaddvq_u32(vandq_u32(vcleq_f32(dist2,limit),lane_bits))<<i;
    }
    return slots;
}

// For the slate shortcut's rays, moving the perpendicular residual to the
// ball-center plane adds at most radius*|dx/dy| (or radius*|dz/dy|) per axis.
// Reserve the rest of the padding for floating-point roundoff.
inline uint32_t packed_overhead_candidates(const PackedBalls& packed,Ray ray,float inverse_y) {
    const float mid=(radius-ray.o.y)*inverse_y;
    const float32x4_t x=vdupq_n_f32(ray.o.x+ray.d.x*mid),z=vdupq_n_f32(ray.o.z+ray.d.z*mid);
    static_assert(radius*max_overhead_lateral_speed/min_overhead_downward_speed < .0005f);
    const float32x4_t limit=vdupq_n_f32(radius+.001f);
    const uint32x4_t lane_bits={1,2,4,8};
    uint32_t slots=0;
    for(int i=0;i<16;i+=4) {
        const float32x4_t dx=vsubq_f32(vld1q_f32(packed.x+i),x);
        const float32x4_t dz=vsubq_f32(vld1q_f32(packed.z+i),z);
        const uint32x4_t mask=vandq_u32(vcaleq_f32(dx,limit),vcaleq_f32(dz,limit));
        slots|=vaddvq_u32(vandq_u32(mask,lane_bits))<<i;
    }
    return slots;
}
#endif

template<bool NeedNormal=true>
ACG_HD inline void intersect_ball_slot(const Balls& balls,Ray r,int slot,float t0,Hit &h) {
#ifdef ACG_COUNT_TRAVERSAL
    ++ball_counters.spheres;
#endif
    float t;
    if(sphere_t(r,balls.c[slot],radius,t0,h.t,t))
        record(h,t,NeedNormal?unit((r.o-balls.c[slot])+r.d*t):Vec{},Ball,slot);
}

struct PocketFrame { Vec center,out,across; float mouth,shelf,half_mouth,flare; };

ACG_HD constexpr PocketFrame pocket_frame(bool corner,float sx,float sz) {
    constexpr float diagonal=.7071067812f;
    if(corner)
        return {{sx*pocket_x,0,sz*pocket_z},{sx*diagonal,0,sz*diagonal},{sx*diagonal,0,-sz*diagonal},
                (table_nose_x+table_nose_z-corner_jaw_setback)*diagonal,.028f,corner_jaw_setback*diagonal,.18f};
    return {{0,0,sz*side_pocket_z},{0,0,sz},{1,0,0},table_nose_z,.004f,side_mouth_half,.25f};
}

ACG_HD inline float pocket_distance2(Vec point,const PocketFrame& frame) {
    const Vec delta=point-frame.center+frame.out*(pocket_tilt*hi(0.f,point.y));
    return delta.x*delta.x+delta.z*delta.z;
}

ACG_HD inline PocketFrame pocket_at(Vec point) {
    return pocket_frame(fabsf(point.x)>pocket_x*.5f,point.x<0?-1.f:1.f,point.z<0?-1.f:1.f);
}
ACG_HD inline float pocket_opening(float y) {
    if(y<pocket_neck_y)
        return pocket_floor_radius+pocket_lower_slope*hi(0.f,y-pocket_floor_y);
    if(y<pocket_seam)
        return pocket_neck_radius+pocket_upper_slope*(y-pocket_neck_y);
    return pocket_throat+pocket_slope*clamp(y-pocket_seam,0.f,pocket_bevel);
}

ACG_HD inline float pocket_distance_to_slot2(Vec point,const PocketFrame& frame) {
    const Vec local=point-frame.center;
    const float u=dot(local,frame.across),v=hi(0.f,dot(local,frame.out)+pocket_tilt*point.y);
    return u*u+v*v;
}
ACG_HD inline bool pocket_cutout(Vec point) {
    const float x=fabsf(point.x),z=fabsf(point.z);
    const bool corner=x>pocket_x*.5f;
    if(fabsf(x-(corner?pocket_x:0.f))>pocket_bound ||
       fabsf(z-(corner?pocket_z:side_pocket_z))>pocket_bound) return false;
    const PocketFrame frame=pocket_frame(corner,point.x<0?-1.f:1.f,point.z<0?-1.f:1.f);
    const float r=pocket_opening(point.y),depth=dot(point,frame.out)-frame.mouth;
    if(point.y<=0) return depth>=frame.shelf && pocket_distance2(point,frame)<r*r;
    return depth>=0 && fabsf(dot(point-frame.center,frame.across))<frame.half_mouth+frame.flare*depth &&
        pocket_distance_to_slot2(point,frame)<r*r;
}
ACG_HD constexpr Vec basis_axis(int k,float s) {
    return k==0?Vec{s,0,0}:k==1?Vec{0,s,0}:Vec{0,0,s};
}
struct Span { float a,b; Vec na,nb; };
struct BoxCrossings { Vec near,far; float entry,exit; };
ACG_HD inline BoxCrossings box_crossings(Box b,Ray r,Vec inv) {
    const Vec p=(b.a-r.o)*inv,q=(b.b-r.o)*inv;
    const Vec l={lo(p.x,q.x),lo(p.y,q.y),lo(p.z,q.z)},u={hi(p.x,q.x),hi(p.y,q.y),hi(p.z,q.z)};
    return {l,u,hi(l.z,hi(l.x,l.y)),lo(u.z,lo(u.x,u.y))};
}
ACG_HD inline bool span_rejected(const BoxCrossings& c,float t0,float t1) {
    return c.exit<c.entry || c.exit<=t0 || c.entry>=t1;
}
template<bool Exit>
ACG_HD inline Vec box_face(Vec t,Vec inv) {
    // Equal crossings choose the Y axis, then X, then Z.
    int k=Exit?(t.x<t.y?0:1):(t.x>t.y?0:1);
    if(Exit?t.z<axis(t,k):t.z>axis(t,k)) k=2;
    const float sign=axis(inv,k)>0?1.f:-1.f;
    return basis_axis(k,Exit?sign:-sign);
}
ACG_HD inline bool box_span(Box b,Ray r,Vec inv,float t0,float t1,Span &s) {
    const BoxCrossings c=box_crossings(b,r,inv);
    if(span_rejected(c,t0,t1)) return false;
    s.a=c.entry; s.b=c.exit;
    s.na=box_face<false>(c.near,inv);
    s.nb=box_face<true>(c.far,inv);
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
ACG_HD constexpr bool is_cushion(Shape shape) { return shape==CushionX || shape==CushionZ; }
ACG_HD constexpr bool is_rail(Shape shape) { return shape==RailX || shape==RailZ; }

ACG_HD inline bool clip_table_part(Span& s,Ray ray,const Prim& part,float margin=0) {
    const bool along_x=part.shape==CushionX || part.shape==RailX;
    const Vec along=along_x?Vec{1,0,0}:Vec{0,0,1};
    const Vec out=along_x?Vec{0,0,float(part.side)}:Vec{float(part.side),0,0};
    const Vec up={0,1,0};
    Vec anchor=(part.box.a+part.box.b)*.5f;
    const auto limit=[&](Vec n,float offset) { return clip(s,ray,n,anchor,offset+margin); };
    if(is_rail(part.shape)) {
        anchor=along_x?Vec{0,rail_top,part.side*rail_z}:Vec{part.side*rail_x,rail_top,0};
        const float miter_end=along_x?rail_x:rail_z;
        return limit(along-out,miter_end) && limit(-along-out,miter_end) &&
            limit(out+up,rail_half_width-.009f) && limit(-out+up,rail_half_width-.0017f);
    }
    anchor=anchor-out*(cushion_width*.5f); anchor.y=cushion_nose_y;
    // Tangents to the nose radius and the cloth ramp to the featherstrip.
    if(!limit(-out+up*.38f,cushion_nose_round*.06977f) ||
       !limit(-out+up*1.18f,cushion_nose_round*.54674f) ||
       !limit(-out-up*.94f,cushion_nose_round*.37244f) ||
       !limit(up-out*cushion_rise,cushion_shoulder)) return false;
    const bool plus_corner=!along_x || anchor.x>0;
    const bool minus_corner=!along_x || anchor.x<0;
    const float tip=(along_x?table_nose_x:table_nose_z)-corner_jaw_setback;
    const float plus_tip=plus_corner?tip:-side_mouth_half;
    const float minus_tip=minus_corner?-tip:side_mouth_half;
    const float center=dot(anchor,along);
    return limit(along-out*(plus_corner?1.28f:.25f),plus_tip-center) &&
        limit(-along-out*(minus_corner?1.28f:.25f),center-minus_tip);
}

// Above the bed the cavity is a cut in the real rail assembly, not a free
// sheet. Reuse the extrusion half-planes for point containment; preserve
// cloth on the cut cushion, and use the inserted liner on the wood backing.
ACG_HD inline bool backed_liner(Vec point,Surface& mat) {
    const StaticScene& scene=room();
    constexpr float margin=.25f*eps;
    bool timber=false;
    for(int i=0;i<scene.size;++i) {
        const Prim& part=scene.p[i];
        const bool cloth=is_cushion(part.shape);
        if(!cloth && !is_rail(part.shape)) continue;
        const Box b=part.box;
        if(point.x<b.a.x-margin || point.x>b.b.x+margin ||
           point.z<b.a.z-margin || point.z>b.b.z+margin) continue;
        Span unused{};
        if(!clip_table_part(unused,{point,{}},part,margin)) continue;
        if(cloth) { mat=CushionCloth; return true; }
        timber=true;
    }
    return timber;
}

template<bool NeedNormal>
ACG_HD inline
void intersect_well(Box b,Ray r,Vec inv,float t0,Hit& h) {
    const PocketFrame frame=pocket_at((b.a+b.b)*.5f);
    const Vec v=r.o-frame.center;
    // Only the slate shelf closes the playing-side face. Above the bed
    // the mouth opens between the two cloth-covered cushion facings.
    const float den=dot(r.d,frame.out);
    if(fabsf(den)>1.e-12f) {
        const float t=(frame.mouth+frame.shelf-dot(r.o,frame.out))/den;
        if(t>t0 && t<h.t) {
            const Vec point=r.o+r.d*t;
            if(point.y>=pocket_floor_y && point.y<=0) {
                const float opening=pocket_opening(point.y);
                if(pocket_distance2(point,frame)<opening*opening)
                    record(h,t,NeedNormal?frame.out:Vec{},Leather);
            }
        }
    }
    const float bottom=(b.a.y-r.o.y)*inv.y;
    if(bottom>t0 && bottom<h.t) {
        const Vec p=v+r.d*bottom;
        if(p.x*p.x+p.z*p.z<=pocket_floor_radius*pocket_floor_radius &&
           dot(r.o+r.d*bottom,frame.out)>=frame.mouth+frame.shelf)
            record(h,bottom,NeedNormal?Vec{0,1,0}:Vec{},Ebony);
    }
    // Close the cut at the flared sides where the jaw is narrower than the slot.
    for(int side=-1;side<=1;side+=2) {
        const Vec normal=frame.across*float(side)-frame.out*frame.flare;
        const float d=dot(normal,r.d);
        if(fabsf(d)<1.e-12f) continue;
        const float boundary=frame.half_mouth+dot(frame.center,frame.across)*side-frame.flare*frame.mouth;
        const float t=(boundary-dot(normal,r.o))/d;
        if(t<=t0 || t>=h.t) continue;
        const Vec point=r.o+r.d*t;
        if(point.y<=0 || point.y>rail_top || dot(point,frame.out)<frame.mouth) continue;
        const float opening=pocket_opening(point.y);
        Surface mat=Leather;
        if(pocket_distance_to_slot2(point,frame)<=opening*opening && backed_liner(point,mat))
            record(h,t,NeedNormal?unit(-normal):Vec{},mat);
    }
    // The two upper stages extend the straight slot into the curved back.
    // Below the bed only the closed, stepped taper remains.
    for(int part=0;part<4;++part) {
        const float y0=part==3?pocket_seam:(part==2?0.f:(part==1?pocket_neck_y:pocket_floor_y));
        const float y1=part==3?pocket_mouth_y:(part==2?pocket_seam:(part==1?0.f:pocket_neck_y));
        const float k=part==3?pocket_slope:(part?pocket_upper_slope:pocket_lower_slope);
        const float section_radius=pocket_opening(y0),r0=section_radius+k*(r.o.y-y0),rd=k*r.d.y;
        const Vec shear=part>=2?frame.out*pocket_tilt:Vec{};
        const Vec o=v+shear*r.o.y,d=r.d+shear*r.d.y;
        if(part>=2) for(int side=-1;side<=1;side+=2) {
            const Vec n=frame.across*float(side);
            const float den=dot(n,r.d)-rd;
            if(fabsf(den)<1.e-12f) continue;
            const float t=(r0-dot(n,v))/den;
            if(t<=t0 || t>=h.t) continue;
            const Vec point=r.o+r.d*t,local=o+d*t;
            const float depth=dot(point,frame.out)-frame.mouth;
            if(point.y<=0 || point.y<y0 || point.y>y1 || depth<0 || dot(local,frame.out)>0 ||
               fabsf(dot(local,frame.across))>frame.half_mouth+frame.flare*depth) continue;
            Surface mat=Leather;
            if(backed_liner(point,mat)) record(h,t,NeedNormal?unit(Vec{-n.x,k,-n.z}):Vec{},mat);
        }
        const float a=d.x*d.x+d.z*d.z-rd*rd;
        const float beta=o.x*d.x+o.z*d.z-r0*rd;
        const float cc=o.x*o.x+o.z*o.z-r0*r0;
        const float det=beta*beta-a*cc;
        if(det<0) continue;
        const float q=-beta-copysignf(sqrtf(det),beta);
        const bool linear=fabsf(a)<1.e-9f;
        for(int j=0;j<2;++j) {
            if((linear && (j || fabsf(beta)<1.e-9f)) || (j && q==0)) continue;
            const float t=linear?-cc/(2*beta):(j?cc/q:q/a);
            if(t<=t0 || t>=h.t) continue;
            const float y=r.o.y+r.d.y*t;
            if(y<y0 || y>y1) continue;
            const Vec point=r.o+r.d*t;
            const float depth=dot(point,frame.out)-frame.mouth;
            if(depth<(y<=0?frame.shelf:0.f)) continue;
            const Vec local=o+d*t;
            Surface mat=Leather;
            if(y>0 && (dot(local,frame.out)<0 || fabsf(dot(local,frame.across))>frame.half_mouth+frame.flare*depth ||
                       !backed_liner(point,mat))) continue;
            if constexpr(NeedNormal) {
                const Vec inward={-local.x,k*(section_radius+k*(y-y0))-dot(local,shear),-local.z};
                record(h,t,unit(inward),mat);
            } else {
                record(h,t,{},mat);
            }
        }
    }
}

template<bool NeedNormal=true>
ACG_HD inline void primitive(const Prim &p,Ray r,Vec inv,float t0,Hit &h) {
    const Box b=p.box;
    if(p.shape==Globe) {
        Vec c=(b.a+b.b)*.5f; float t;
        if(sphere_t(r,c,(b.b.x-b.a.x)*.5f,t0,h.t,t))
            record(h,t,NeedNormal?unit((r.o-c)+r.d*t):Vec{},Glass);
        return;
    }
    if(p.shape==Screen) {
        if(fabsf(r.d.z)<1.e-12f) return;
        float t=(b.b.z-r.o.z)*inv.z;
        if(t>t0 && t<h.t) {
            Vec v=r.o+r.d*t;
            if(v.x>=b.a.x && v.x<=b.b.x && v.y>=b.a.y && v.y<=b.b.y)
                record(h,t,NeedNormal?Vec{0,0,1}:Vec{},Grille);
        }
        return;
    }
#ifdef ACG_PACKED_ROOM
    const BoxCrossings crossings=box_crossings(b,r,inv);
    if(span_rejected(crossings,t0,h.t)) return;
#else
    Span s;
    if(!box_span(b,r,inv,t0,h.t,s)) return;
#endif
    if(p.shape==Well) {
        intersect_well<NeedNormal>(b,r,inv,t0,h);
        return;
    }
#ifdef ACG_PACKED_ROOM
    Span s={crossings.entry,crossings.exit,{},{}};
#endif
    const bool cushion=is_cushion(p.shape),rail=is_rail(p.shape);
    if(cushion || rail) {
#ifdef ACG_PACKED_ROOM
        if constexpr(NeedNormal) {
            s.na=box_face<false>(crossings.near,inv);
            s.nb=box_face<true>(crossings.far,inv);
        }
#endif
        if(!clip_table_part(s,r,p)) return;
    }
    for(int k=0;k<2;++k) {
        float t=k?s.b:s.a; if(t<=t0 || t>=h.t) continue;
        if(p.shape==BoredBlock || cushion || rail) {
            Vec point=r.o+r.d*t;
            // Reconstructing a top hit may round into the upper mouth.
            if(p.shape==BoredBlock) point.y=lo(point.y,b.b.y);
            if(pocket_cutout(point)) continue;
        }
#ifdef ACG_PACKED_ROOM
        const Vec n=NeedNormal?((cushion||rail)?unit(k?s.nb:s.na):
            (k?box_face<true>(crossings.far,inv):box_face<false>(crossings.near,inv))):Vec{};
        record(h,t,n,p.mat);
        break;
#else
        Vec n=k?s.nb:s.na; record(h,t,NeedNormal?((cushion||rail)?unit(n):n):Vec{},p.mat); break;
#endif
    }
}

#ifndef ACG_LIGHT_HALF_WIDTH
#define ACG_LIGHT_HALF_WIDTH .30f
#endif
constexpr float light_y=1.37f,light_z=-.52f,light_hx=.79f,light_hz=ACG_LIGHT_HALF_WIDTH;
constexpr float camera_height=light_y-.12f;
constexpr Vec point_light_position={-.73f,1.54f,-.46f};

// The central column between y=0 and the given height contains just the slate.
// Its Z margin from either pocket row exceeds the largest pocket opening.
constexpr bool clear_slate_column(float height) {
    if(!(table_clear_z+pocket_bound+eps<lo(pocket_z,side_pocket_z))) return false;
    bool have_slate=false;
    for(int i=0;i<host_room.size;++i) {
        const Prim& p=host_room.p[i];
        if(p.shape==Globe) return false;
        const Box b=p.box;
        if(b.b.y< -eps || b.a.y>height ||
           b.b.x< -table_clear_x || b.a.x>table_clear_x ||
           b.b.z< -table_clear_z || b.a.z>table_clear_z) continue;
        if(have_slate || p.shape!=BoredBlock || p.mat!=Felt || p.side==2 ||
           b.a.y>=0 || b.b.y!=0 || b.a.x>=-table_clear_x || b.b.x<=table_clear_x ||
           b.a.z>=-table_clear_z || b.b.z<=table_clear_z) return false;
        have_slate=true;
    }
    return have_slate;
}
static constexpr bool overhead_slate_clear=clear_slate_column(camera_height);
static_assert(light_hz>0,"Emitter half width must be positive");
constexpr float light_area=4*light_hx*light_hz;
ACG_HD constexpr Vec emission() { return {6.7f,6.1f,5.3f}; }
#include "soft_intervals.h"
struct ScalarSlabs {
    Ray ray;
    Vec inverse;
    float t0;

    ACG_HD inline bool operator()(const Node& node,float t1) const {
        return overlaps(node.box,ray,inverse,t0,t1);
    }
};

struct NodeIndices { int first,count,escape; };

template<class TreeNode>
ACG_HD inline NodeIndices node_indices(const TreeNode& node) {
    return {node.first,node.count,node.escape};
}

#ifdef ACG_PACKED_ROOM
inline NodeIndices node_indices(const PackedNode& node) {
    uint32_t bits;
    std::memcpy(&bits,&node.lower[3],sizeof(bits));
    return {int(bits&packed_index_mask),int((bits>>packed_index_bits)&packed_index_mask),
            int((bits>>(2*packed_index_bits))&packed_index_mask)};
}
#endif

template<bool NeedNormal=true,class TreeNode,class Slabs>
ACG_HD inline void intersect_static(const Prim* primitives,const TreeNode* nodes,int count,
                                    Ray ray,Vec inv,int stage,float t0,Hit &h,const Slabs& slabs) {
    for(int i=0;i<count;) {
        const TreeNode& node=nodes[i];
        const NodeIndices indices=node_indices(node);
        if(!slabs(node,h.t)) { i=indices.escape; continue; }
        for(int j=indices.first;j<indices.first+indices.count;++j) {
            const Prim& p=primitives[j];
            if(stage!=0 || p.side!=2) primitive<NeedNormal>(p,ray,inv,t0,h);
        }
        ++i;
    }
}

template<bool NeedNormal=true,bool Overhead=false>
ACG_HD inline void intersect_balls(Ray r,Vec inv,const Balls &balls,float t0,Hit &h) {
#ifdef ACG_COUNT_TRAVERSAL
    ++ball_counters.rays;
    if(!balls.tree) ++ball_counters.boxes;
#endif
    if(balls.tree) {
        intersect_ball_bvh<NeedNormal>(*balls.tree,balls,r,inv,t0,h);
    } else if(overlaps(balls.box,r,inv,t0,h.t)) {
#ifdef ACG_PACKED_ROOM
        if(balls.packed && (Overhead || packed_ball_ray(r))) {
            const uint32_t candidates=Overhead?packed_overhead_candidates(*balls.packed,r,inv.y):
                                               packed_ball_candidates(*balls.packed,r);
            for(uint32_t slots=candidates;slots;slots&=slots-1)
                intersect_ball_slot<NeedNormal>(balls,r,__builtin_ctz(slots),t0,h);
        } else
#endif
        for(int i=0;i<16;++i) intersect_ball_slot<NeedNormal>(balls,r,i,t0,h);
    }
}

ACG_HD inline float emitter_hit(Ray r,float inverse_y,float t0,float t1) {
    if(fabsf(r.d.y)<=1.e-12f) return t1;
    const float t=(light_y-r.o.y)*inverse_y;
    if(!(t>t0 && t<t1)) return t1;
    const Vec p=r.o+r.d*t;
    return fabsf(p.x)<light_hx && fabsf(p.z-light_z)<light_hz?t:t1;
}

// Visibility needs the same nearest hit and material, but never reads its normal.
template<bool NeedNormal=true>
ACG_HD inline bool intersect_with_inverse(Ray r,Vec inv,const Balls &balls,int stage,float t0,float t1,
                                          Hit &h,bool interior=false,bool skip_static=false) {
    h.t=t1;
    intersect_balls<NeedNormal>(r,inv,balls,t0,h);
    if(skip_static) return h.t<t1;
    const StaticScene &s=room();
#if defined(ACG_PACKED_ROOM)
    if(!isnan(t0) && !isnan(t1) && (interior || packed_full_ray(r.o,inv))) {
        const PackedSlabs slabs{r.o,inv,t0};
        if(interior && inv.y>0 && t0>0 &&
           (r.o.y>=pocket_top || clears_table_center(r.o,inv,h.t))) {
            intersect_static<NeedNormal>(s.p,packed_above_rails.data(),static_cast<int>(packed_above_rails.size()),
                             r,inv,stage,t0,h,slabs);
        } else {
            const PackedNode* nodes=interior?packed_interior.data():packed_room.data();
            const int count=interior?static_cast<int>(packed_interior.size()):static_cast<int>(packed_room.size());
            intersect_static<NeedNormal>(s.p,nodes,count,r,inv,stage,t0,h,slabs);
        }
    } else {
        const Node* nodes=interior?interior_tree.n:s.n;
        const int count=interior?interior_tree.size:s.nodes;
        intersect_static<NeedNormal>(s.p,nodes,count,r,inv,stage,t0,h,ScalarSlabs{r,inv,t0});
    }
#elif !defined(__CUDA_ARCH__)
    const Node* nodes=interior?interior_tree.n:s.n;
    const int count=interior?interior_tree.size:s.nodes;
    intersect_static<NeedNormal>(s.p,nodes,count,r,inv,stage,t0,h,ScalarSlabs{r,inv,t0});
#else
    intersect_static<NeedNormal>(s.p,s.n,s.nodes,r,inv,stage,t0,h,ScalarSlabs{r,inv,t0});
#endif
    if(stage==2) {
        const float t=emitter_hit(r,inv.y,t0,h.t);
        if(t<h.t) record(h,t,NeedNormal?Vec{0,-1,0}:Vec{},Lamp);
    }
    return h.t<t1;
}

ACG_HD inline bool intersect(Ray r,const Balls &balls,int stage,float t0,float t1,Hit &h) {
    const Vec inv=invdir(r.d);
#if !defined(__CUDACC__) && !defined(__CUDA__)
    if constexpr(overhead_slate_clear) {
        constexpr float inset=.02f;
        static_assert(camera_height*max_overhead_lateral_speed/min_overhead_downward_speed < inset);
        if(r.d.y< -min_overhead_downward_speed && r.d.y>=-1.01f &&
           r.o.y>0 && r.o.y<=camera_height && t0>0) {
            const float depth=(0.f-r.o.y)*inv.y;
            // At this height and slope the ray travels less than 2 cm in X/Z.
            const bool clear=fabsf(r.d.x)<=max_overhead_lateral_speed &&
                fabsf(r.d.z)<=max_overhead_lateral_speed &&
                fabsf(r.o.x)<table_clear_x-inset && fabsf(r.o.z)<table_clear_z-inset;
            if(clear && depth>t0 && depth<t1) {
                h.t=t1;
                intersect_balls<true,true>(r,inv,balls,t0,h);
                // Balls keep priority when their hit is exactly at slate depth.
                if(depth<h.t) record(h,depth,{0,1,0},Felt);
                return true;
            }
        }
    }
#endif
    return intersect_with_inverse(r,inv,balls,stage,t0,t1,h);
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
// Roll off periodic detail before its phase spans a full pixel cycle.
ACG_HD constexpr float texture_band(float phase_halfwidth) {
    const float t=clamp((pi-phase_halfwidth)*(1.5f/pi),0,1);
    return t*t*(3-2*t);
}

ACG_HD inline float grain(Vec p,float footprint=0) {
    float bend=2.8f*sinf(7.1f*p.x+2.9f*p.z)+1.5f*sinf(17.3f*p.x+7*p.y);
    const float rings=.27f*texture_band(150*footprint)*sinf(96*p.z+59*p.y+bend);
    const float lines=.12f*texture_band(330*footprint)*sinf(239*p.z+123*p.y+2.3f*bend);
    const float pores=.06f*texture_band(660*footprint)*sinf(611*p.z+2*bend);
    return clamp(.53f+rings+lines+pores,0,1);
}

ACG_HD inline float cloth_finish(float along,float across) {
    // Use a visible weave scale at the replay camera's sampling density.
    const float nap=sinf(39*along+17*across+.8f*sinf(12*across));
    const float weave=sinf(875*along+22*across)*sinf(760*across);
    return .955f+.018f*nap+.036f*weave;
}

ACG_HD inline float rail_grain(Vec p) {
    // Longitudinal figure, narrow growth bands and finer vessel streaks.
    const float bend=.005f*sinf(5*p.x+2*p.z)+.0015f*sinf(27*p.x+9*p.y);
    const float cross=p.z+.4f*p.y+bend;
    const float phase=274*cross+1.4f*sinf(109*cross+5*p.x);
    const float latewood=.5f+.5f*sinf(phase);
    const float figure=sinf(17*p.x+85*cross)*sinf(47*cross-4*p.x);
    const float pores=sinf(572*cross+7*p.x)*sinf(84*p.x+66*cross);
    return clamp(.60f-.30f*latewood*latewood+.095f*figure+.066f*pores,0,1);
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
    if(num==0)
        return hi(fabsf(n.x),hi(fabsf(n.y),fabsf(n.z)))>.994f?Vec{.48f,.046f,.041f}:body;
    // Longitude/latitude charts give two antipodal number badges and the belt.
    float lon=atan2f(n.x,n.z),lat=asinf(clamp(n.y,-1,1));
    if(lon>pi*.5f) lon-=pi; if(lon<-pi*.5f) lon+=pi;
    Vec ivory={.90f,.835f,.686f};
    Vec c=num>8 && fabsf(lat)>.49f?ivory:body;
    const float rr=lon*lon+lat*lat;
    if(rr<.45f*.45f) {
        c=rr>.412f*.412f?mix(body,grey(.025f),.48f):ivory;
        const float u=lon/.092f,v=2.5f-lat/.095f;
        const bool ink=num<10?digit(num,u+1.5f,v):(digit(1,u+3.35f,v)||digit(num%10,u-.35f,v));
        if(ink) c={.013f,.017f,.022f};
    }
    return c;
}

ACG_HD inline float marble_vein(float phase,float footprint) {
    float vein=1-fabsf(sinf(phase));
    vein*=vein; vein*=vein; vein*=vein;
    constexpr float mean=.0714212f; // Integral of (1 - |sin(x)|)^8 over one period.
    return mean+(vein-mean)*texture_band(68*footprint);
}

struct Material { Vec diffuse,f0; float rough; };
ACG_HD inline Material varnish(Vec c,float rough) { return {c,grey(.044f),rough}; }
ACG_HD inline Material material(Hit h,Vec p,const Balls &balls,float t,int stage,float footprint=0) {
    if(stage==0) {
        Vec c;
        switch(h.mat) {
            case Felt:c={.035f,.30f,.175f};break; case Ball:c=pigment(number_for(h.id));break;
            case CushionCloth:c={.027f,.21f,.13f};break; case Leather:c={.052f,.043f,.036f};break;
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
                base=mix({.112f,.031f,.012f},{.64f,.301f,.088f},grain(n*.53f,footprint*.53f)); rough=.205f;
            }
            if(num==4) { // Veined marble, evaluated in ball-local coordinates.
                float bend=.21f*sinf(8*n.z)+.09f*sinf(19*n.x+11*n.z);
                const float vein=marble_vein(13*(n.y+bend),footprint);
                base=mix({.76f,.73f,.66f},{.14f,.18f,.23f},vein); rough=.23f;
            } else if(num==6) { // Layered ceramic glaze with mottled pigment.
                float mottling=.5f+.5f*sinf(17*n.x+3*sinf(13*n.z))*sinf(19*n.y);
                base=mix({.016f,.085f,.19f},{.16f,.58f,.66f},mottling); rough=.18f;
            } else if(num==7) { // Keep weave only where the projected thread pitch resolves.
                constexpr Vec average={.27675f,.1465f,.0745f};
                const float band=texture_band(48*footprint/hi(.18f,sqrtf(hi(0,1-n.y*n.y))));
                base=average;
                if(band>0) {
                    const float a=atan2f(n.x,n.z)*(12/pi),b=asinf(clamp(n.y,-1,1))*(12/pi);
                    const float warp=1-fabsf(2*fract(a)-1),weft=1-fabsf(2*fract(b)-1);
                    const bool over=fract((floorf(a)+floorf(b))*.5f)<.25f;
                    const Vec woven=over?mix({.083f,.029f,.017f},{.58f,.28f,.15f},warp)
                                        :mix({.064f,.037f,.021f},{.38f,.24f,.11f},weft);
                    base=mix(average,woven,band);
                }
                rough=.46f;
            }
            // Camera samples already integrate these sharp graphic marks.
            return varnish(enamel(num,n,base),rough);
        }
        case Felt: {
            const float tone=cloth_finish(p.x,p.z);
            return {Vec{.027f,.248f,.154f}*tone,grey(.019f),.82f+.10f*(tone-.955f)};
        }
        case CushionCloth: {
            const bool end=fabsf(p.x)>1.25f && fabsf(p.z)<.59f;
            const float along=end?p.z:p.x;
            const float cross_rail=end?fabsf(p.x)-table_nose_x:fabsf(p.z)-table_nose_z;
            const float weave=cloth_finish(along,cross_rail+.30f*p.y);
            const float binding=1-.10f*hi(0,1-fabsf(cross_rail-featherstrip)/.0016f);
            return {Vec{.024f,.214f,.133f}*(weave*binding),grey(.021f),.88f};
        }
        case Timber: {
            const float joint=(fabsf(p.x)-rail_x)-(fabsf(p.z)-rail_z);
            const bool end=joint>0;
            const Vec wood=end?Vec{p.z+.27f,p.y,p.x}:p;
            Vec c=mix({.081f,.029f,.011f},{.276f,.122f,.044f},rail_grain(wood));
            const float across=end?fabsf(p.x)-rail_x:fabsf(p.z)-rail_z;
            // A narrow dark inlay follows the grain along the outer edge.
            c=c*(1-.36f*hi(0,1-fabsf(across-.031f)/.0028f));
            if(fabsf(p.x)>=rail_x-rail_half_width && fabsf(p.z)>=rail_z-rail_half_width)
                c=c*(1-.28f*hi(0,1-fabsf(joint)/.0018f));
            if(h.normal.y>.95f && fabsf(p.y-rail_top)<.001f) {
                const float along=end?p.z:p.x,mark=across+featherstrip+rail_half_width-.0936625f;
                const float index=floorf(along/.3175f+.5f);
                const float d=fabsf(along-index*.3175f);
                const float diamond=d/.010f+fabsf(mark)/.0065f;
                const bool has_sight=end?fabsf(index)<=1:(index!=0 && fabsf(index)<=3);
                if(has_sight && diamond<1 && pocket_distance2(p,pocket_at(p))>pocket_bound*pocket_bound)
                    c=mix(c,{.71f,.66f,.53f},clamp((1-diamond)*9,0,1));
            }
            return varnish(c,.29f);
        }
        case Brass:case Grille:return {grey(0),{.83f,.592f,.283f},h.mat==Grille?.31f:.235f};
        case Ebony: {
            const float weave=.84f+.16f*sinf(820*p.x)*sinf(730*p.z);
            return varnish(Vec{.039f,.031f,.024f}*weave,.86f);
        }
        case Leather: {
            const float pebble=.91f+.06f*sinf(610*p.x+310*p.y)*sinf(570*p.z-240*p.y);
            const float seam=hi(0,1-fabsf(p.y-pocket_neck_y)/.002f);
            const Vec leather=p.y>=pocket_neck_y?Vec{.058f,.047f,.039f}:Vec{.036f,.028f,.023f};
            return varnish(leather*(pebble*(1-.21f*seam)),.74f);
        }
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
    // Expand both cross products using the zero component of the tangent.
    if(fabsf(z.y)<.92f) {
        const Vec x=unit({z.z,0,-z.x});
        return {x,{z.y*x.z,z.z*x.x-z.x*x.z,-z.y*x.x},z};
    }
    const Vec x=unit({-z.y,z.x,0});
    return {x,{-z.z*x.y,z.z*x.x,z.x*x.y-z.y*x.x},z};
}
ACG_HD inline Vec world(Frame f,Vec a) { return f.x*a.x+f.y*a.y+f.z*a.z; }
ACG_HD inline Vec local(Frame f,Vec a) { return {dot(a,f.x),dot(a,f.y),dot(a,f.z)}; }
ACG_HD inline Vec fresnel(Vec f0,float c) {
    float x=clamp(1-c,0,1),x2=x*x; return f0+(grey(1)-f0)*(x2*x2*x);
}
ACG_HD inline float spec_prob(Material m) {
    const float d=peak(m.diffuse),s=peak(m.f0);
    if(s==0) return 0;
    if(d==0) return 1;
    constexpr float min_probability=.16f;
    // If d >= 8*s > 0, the ratio is at most 1/8, below the clamp floor.
    if(s>0 && 8.f*s<=d) return min_probability;
    return clamp(s/(s+d),min_probability,.8f);
}
ACG_HD inline float smith_g1(float c,float a2) { return 2*c/(c+sqrtf(a2+(1-a2)*c*c)); }
// GGX terms shared by all direct-light samples and the continuation at a hit.
struct BrdfView {
    float cosine, specular_probability, alpha_squared, masking;
};
ACG_HD inline BrdfView prepare_brdf_view(Material m,Vec n,Vec wo) {
    const float cosine=dot(n,wo);
    const float specular_probability=spec_prob(m);
    const float alpha=m.rough*m.rough,alpha_squared=alpha*alpha;
    const float masking=cosine<=0 || specular_probability==0?0.f:smith_g1(cosine,alpha_squared);
    return {cosine,specular_probability,alpha_squared,masking};
}
// The GGX sampler draws visible half-normals, so the specular reflection
// density in solid angle is D(h) G1(wo) / (4 n.wo).
ACG_HD inline Vec brdf(Material m,Vec n,Vec wo,Vec wi,const BrdfView& view,float &pdf) {
    const float a=view.cosine,b=dot(n,wi);
    pdf=0; if(a<=0 || b<=0) return grey(0);
    const float ps=view.specular_probability;
    pdf=(1-ps)*b/pi;
    if(ps==0) return m.diffuse*(1/pi);
    const Vec h=unit(wo+wi); const float nh=clamp(dot(n,h),0,1);
    const float a2=view.alpha_squared;
    const float d=(1-nh*nh)+nh*nh*a2,D=a2/(pi*d*d),g=view.masking;
    pdf+=ps*D*g/(4*a);
    const Vec F=fresnel(m.f0,dot(wi,h));
    return (grey(1)-F)*m.diffuse*(1/pi)+F*(D*g*smith_g1(b,a2)/(4*a*b));
}

ACG_HD inline Vec sample_direction(Vec n,Vec wo,float rough,float specular_probability,Rng &rng) {
    Frame f=frame(n); float choice=rng.next(),u=rng.next(),v=rng.next();
    float rad=sqrtf(u),ang=2*pi*v,dx=rad*cosf(ang),dy=rad*sinf(ang);
    if(choice>=specular_probability) return world(f,{dx,dy,sqrtf(1-u)});
    float a=rough*rough; Vec view=local(f,wo),vh=unit({a*view.x,a*view.y,view.z});
    float l=vh.x*vh.x+vh.y*vh.y;
    Vec t1=l>1.e-15f?Vec{-vh.y,vh.x,0}*(1/sqrtf(l)):Vec{1,0,0};
    float s=.5f*(1+vh.z); dy=(1-s)*sqrtf(hi(0,1-dx*dx))+s*dy;
    Vec h=t1*dx+cross(vh,t1)*dy+vh*sqrtf(hi(0,1-dx*dx-dy*dy));
    Vec wh=world(f,unit({a*h.x,a*h.y,hi(0,h.z)})); return unit(wh*(2*dot(wo,wh))-wo);
}
ACG_HD inline float power_weight(float a,float b) { return a*a/(a*a+b*b); }
ACG_HD inline float visibility(Ray r,float limit,const Balls &balls,int stage,bool skip_static=false) {
    Hit h;
    const Vec inv=invdir(r.d);
#if !defined(__CUDA_ARCH__)
    const bool interior=!skip_static && inside_room(r.o,inv,limit);
#else
    constexpr bool interior=false;
#endif
    if(!intersect_with_inverse<false>(r,inv,balls,stage,eps,limit,h,interior,skip_static)) return 1;
    if(h.mat!=Grille) return 0; // A thick refractor is not a straight shadow path.
    float through=1-coverage(r.o+r.d*h.t);
    while(through!=0 && intersect_with_inverse<false>(r,inv,balls,stage,h.t+eps,limit,h,interior)) {
        if(h.mat!=Grille) return 0;
        through*=1-coverage(r.o+r.d*h.t);
    }
    return through;
}

#if !defined(__CUDACC__) && !defined(__CUDA__)
constexpr float shadow_column_margin=.001f;
constexpr bool shadow_column_clear=clear_slate_column(point_light_position.y+shadow_column_margin);
#endif

ACG_HD inline float visibility_to_light(Vec origin,Vec source,const Balls &balls,int stage) {
    const Vec delta=source-origin;
    const float distance=sqrtf(dot(delta,delta));
    const Ray ray={origin,delta*(1/distance)};
    const float limit=distance-3*eps;
    // When the opaque fixture is between a receiver and the room fill,
    // it settles visibility without traversing the ball or room BVH.
    if(stage==2 && source.y>light_y && origin.y<light_y &&
       emitter_hit(ray,recip(ray.d.y),eps,limit)<limit) return 0;
    bool skip_static=false;
#if !defined(__CUDACC__) && !defined(__CUDA__)
    if constexpr(shadow_column_clear) {
        // Both ends lie inside the clear column. The inset covers rounding in
        // the normalized direction and slab comparisons.
        constexpr float x=table_clear_x-shadow_column_margin,z=table_clear_z-shadow_column_margin;
        // A segment ending exactly in the emitter plane uses normal traversal.
        // The other upward segments either cannot reach it or were tested above.
        skip_static=(stage!=2 || source.y!=light_y) && fabsf(source.z)<z && fabsf(origin.z)<z &&
            fabsf(source.x)<x && fabsf(origin.x)<x &&
            origin.y>0 && source.y>origin.y && source.y<=point_light_position.y && limit>=eps;
    }
#endif
    return visibility(ray,limit,balls,stage,skip_static);
}
#ifndef ACG_AREA_SAMPLES
#define ACG_AREA_SAMPLES 2
#endif
static_assert((ACG_AREA_SAMPLES)==1 || (ACG_AREA_SAMPLES)==2,
              "Expected one or two area-light samples");
constexpr int area_samples=ACG_AREA_SAMPLES;

enum class FeltPass { Full, Basis, Residual };

// View-independent first-hit diffuse basis. Keep Schlick's directional
// remainder, including any negative values, in the local estimator.
ACG_HD constexpr Vec felt_diffuse_basis(Material m) {
    return m.diffuse*(grey(1)-m.f0);
}

template<FeltPass Pass>
ACG_HD inline Vec area_brdf(Vec f,Material m) {
    if constexpr (Pass==FeltPass::Basis) return grey(1/pi);
    if constexpr (Pass==FeltPass::Residual) return f-felt_diffuse_basis(m)*(1/pi);
    return f;
}

template<FeltPass Pass=FeltPass::Full>
ACG_HD inline Vec direct_sample(Vec p,Vec n,Vec wo,Vec org,Vec source,Vec intensity,bool area,
                                Material m,const BrdfView& view,const Balls &balls,int stage,float cached,bool compete) {
    Vec delta=source-p; float d2=dot(delta,delta),dist=sqrtf(d2); Vec wi=delta*(1/dist);
    float c=dot(n,wi); if(c<=0 || (area && wi.y<=0)) return grey(0);
    // Aim from the offset origin to the actual sample (including near grazing).
    float vis=cached>=0?cached:visibility_to_light(org,source,balls,stage); if(vis==0) return grey(0);
    float pdf; const Vec f=brdf(m,n,wo,wi,view,pdf);
    if(!area) return f*intensity*(vis*c/d2); // radiant intensity, inverse square
    float lp=d2/(wi.y*light_area);
    return area_brdf<Pass>(f,m)*intensity*(vis*c*(compete?power_weight(area_samples*lp,pdf):1.f)/(lp*area_samples));
}
#ifdef __CUDACC__
ACG_HD __noinline__
#else
inline
#endif
bool analytic_area_sample(Vec p,Vec n,Vec wo,Vec q,Material m,const BrdfView& view,const Balls& balls,Vec& result,bool compete) {
    SoftSlice slice;
    if(!soft_slice(p,balls.c,q.z,slice,soft_lamp,radius,eps)) return false;
    Vec c=m.diffuse*(1/pi);
    Vec sum=c*float(slice.geometry);
    if(soft_visible(slice,q.x)) {
        Vec d=q-p; float d2=dot(d,d); Vec wi=d*(1/sqrtf(d2));
        float pdf; Vec f=brdf(m,n,wo,wi,view,pdf);
        float lp=d2/(wi.y*light_area);
        sum=sum+(f*(compete?power_weight(area_samples*lp,pdf):1.f)-c)*(2*light_hx*wi.y*wi.y/d2);
    }
    result=emission()*sum*(2*light_hz/area_samples);
    return true;
}
struct FirstLight {
    uint32_t sample=~0u,key=0;
};

ACG_HD inline Vec first_light_r2(FirstLight first) {
    // Shift the R2 sequence independently per pixel in integer coordinates.
    // Unlike s/spp strata, prefixes remain nested when the budget changes.
    const uint32_t u=hash32(first.key^0xa4aba1c3u)+0xc13fa9a9u*first.sample;
    const uint32_t v=hash32(first.key^0x392fe91bu)+0x91e10da5u*first.sample;
    return {float(u>>8)*0x1p-24f,0,float(v>>8)*0x1p-24f};
}

template<FeltPass Pass=FeltPass::Full>
ACG_HD inline Vec lighting(Vec p,Vec n,Vec wo,Material m,const BrdfView& view,const Balls &balls,int stage,
                           Rng &rng,const Hit& hit,bool analytic,float cached,bool compete,FirstLight first={}) {
    // Stage 2 uses a large key over the table and a weak fill. Stage 0/1
    // retain the unmodified point-light diagnostic.
    Vec I=stage==2?Vec{.36f,.326f,.304f}:Vec{8.05f,7.35f,6.37f};
    const Vec origin=p+n*eps;
    Vec out{};
    if constexpr (Pass!=FeltPass::Basis)
        out=direct_sample(p,n,wo,origin,point_light_position,I,false,m,view,balls,stage,cached,false);
    if(stage==2 && p.y<light_y) {
        const bool eligible=Pass==FeltPass::Full && analytic && soft_eligible(stage,hit,p,n,balls.c);
        const bool r2=first.sample!=~0u;
        const Vec stratum=r2?first_light_r2(first):Vec{};
#ifdef ACG_STRATIFY_LIGHT_2D
        // Randomly pair the X and Z strata; a fixed diagonal would leave half
        // of the emitter unsampled. The marginal light PDF is unchanged.
        static_assert(area_samples==2,"two strata per axis");
        const int flip=int(hash32(rng.s ^ 0x640f32d1u)>>31);
#endif
        for(int i=0;i<area_samples;++i) {
            float jx=rng.next(),v=rng.next();
            if(r2) {
                const float shift=float(i)/area_samples;
                jx=eligible?fract(stratum.x+shift):stratum.x;
                v=fract(stratum.z+shift);
            }
#ifdef ACG_STRATIFY_LIGHT_2D
            if(!eligible && !r2) v=((i^flip)+v)*.5f;
#endif
            const float u=eligible?jx:(i+jx)/area_samples;
            Vec q={(2*u-1)*light_hx,light_y,light_z+(2*v-1)*light_hz};
            Vec contribution;
            if(eligible && analytic_area_sample(p,n,wo,q,m,view,balls,contribution,compete)) {
                out=out+contribution;
                continue;
            }
            // Only the certified felt passes may reuse full lamp visibility.
            constexpr float vis=Pass==FeltPass::Full?-1.f:1.f;
            out=out+direct_sample<Pass>(p,n,wo,origin,q,emission(),true,m,view,balls,stage,vis,compete);
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
template<FeltPass Pass>
ACG_HD inline Vec trace_felt_pass(Ray ray,const Balls &balls,int stage,float t,Rng &rng,
                                  bool analytic=false,float cached=-1,Vec irradiance={},float pixel_width=0,
                                  FirstLight first={}) {
    Vec out={},beta=grey(1),prev={}; float prev_pdf=0,eta_scale=1; bool delta=true,inside=false;
#ifndef ACG_MAX_HITS
#define ACG_MAX_HITS 14
#endif
    constexpr int max_hits=ACG_MAX_HITS;
    static_assert(max_hits>=1 && max_hits<=14,"Invalid preview path depth");
    for(int depth=0;depth<max_hits;++depth) {
        Hit hit; float t0=eps;
        const bool felt=Pass!=FeltPass::Full && depth==0;
        // The caller certifies the entire camera footprint on the top slate.
        if(felt) record(hit,-ray.o.y/ray.d.y,{0,1,0},Felt);
        else for(;;) {
            if(!intersect(ray,balls,stage,t0,far_t,hit)) return out;
            if(hit.mat!=Grille || rng.next()<coverage(ray.o+ray.d*hit.t)) break;
            t0=hit.t+eps;
        }
        Vec p=ray.o+ray.d*hit.t,wo=-ray.d;
        bool front=dot(wo,hit.normal)>0; Vec n=front?hit.normal:-hit.normal;
        if(room_has_glass && inside) beta=beta*Vec{expf(-1.63f*hit.t),expf(-.24f*hit.t),expf(-.45f*hit.t)};
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
        if(room_has_glass && hit.mat==Glass) {
#ifdef ACG_NO_REFRACTION
            wi=ray.d-n*(2*dot(ray.d,n));
#else
            float w; bool transmit=glass_bounce(ray.d,n,front,rng,wi,w);
            if(transmit) { inside=front; beta=beta*w; eta_scale/=w; }
#endif
            delta=true;
        } else {
            const float footprint=depth==0 && hit.mat==Ball?
                lo(.30f,(.5f*pixel_width/radius)/hi(.14f,dot(n,wo))):0.f;
            Material m=material(hit,p,balls,t,stage,footprint);
            const float vis=depth==0 && hit.mat==Felt && n.y==1 && fabsf(p.y)<eps?cached:-1.f;
            const bool continue_path=stage!=0 && depth+1<max_hits;
            const BrdfView view=prepare_brdf_view(m,n,wo);
            if(felt)
                out=out+lighting<Pass>(p,n,wo,m,view,balls,stage,rng,hit,false,-1.f,continue_path,first);
            else
                out=out+beta*lighting(p,n,wo,m,view,balls,stage,rng,hit,analytic,vis,continue_path,
                                      depth==0?first:FirstLight{});
            if constexpr (Pass==FeltPass::Residual)
                if(felt) out=out+irradiance*felt_diffuse_basis(m);
            if(!continue_path) break;
            wi=sample_direction(n,wo,m.rough,view.specular_probability,rng); float c=dot(wi,n);
            if(c<=0) break;
            Vec f=brdf(m,n,wo,wi,view,prev_pdf); if(prev_pdf<=0) break;
            // Keep the same sampling PDF and both sides of the emitter MIS
            // partition for the basis and signed residual.
            if(felt) f=area_brdf<Pass>(f,m);
            beta=beta*f*(c/prev_pdf); prev=p; delta=false;
        }
        if(depth>=3) {
            float q=clamp(peak(beta)*eta_scale,.08f,.95f); if(rng.next()>=q) break; beta=beta*(1/q);
        }
        ray={p+n*(dot(wi,n)>0?eps:-eps),wi};
    }
    return out;
}
ACG_HD inline Vec trace(Ray ray,const Balls &balls,int stage,float t,Rng &rng,bool analytic=false,float cached=-1,
                       float pixel_width=0,FirstLight first={}) {
    return trace_felt_pass<FeltPass::Full>(ray,balls,stage,t,rng,analytic,cached,{},pixel_width,first);
}
ACG_HD constexpr float camera_half_height(float aspect) { return hi(.865f,1.54f/aspect); }

ACG_HD constexpr float camera_pixel_width(int w,int h) {
    return 2*camera_half_height(float(w)/h)/h;
}
ACG_HD constexpr Vec camera_plane(float x,float y,int w,int h,float pitch) {
    return {(x-.5f*w)*pitch,0,(y-.5f*h)*pitch};
}
ACG_HD inline Ray camera(int x,int y,int w,int h,int stage,Rng &rng) {
    const float pitch=camera_pixel_width(w,h);
    float jx=rng.next(), jy=rng.next();
#ifdef ACG_NO_AA
    jx=jy=.5f;
#endif
    Vec origin=camera_plane(float(x)+jx,float(y)+jy,w,h,pitch);
    origin.y=camera_height;
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
    Ray r=camera(x,y,w,h,stage,rng); return trace(r,poses,stage,t,rng,analytic,cached,camera_pixel_width(w,h));
}
#endif
