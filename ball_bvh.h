#ifndef ACG_BALL_BVH_H_INCLUDED
#define ACG_BALL_BVH_H_INCLUDED

// Include inside acg_scene_detail after sphere_t. Build on the host, then
// upload the tree alongside the matching Balls snapshot. No host pointers.
struct BallBVH {
    static constexpr int slot_count=16, node_count=2*slot_count-1;
    Box box[node_count]; // preorder; the root also serves as the broad phase
    int8_t tag[node_count]; // leaf: original slot; internal: negated subtree end (exclusive)
};

namespace ball_bvh_detail {
inline Box bounds(Vec c) {
    const Vec pad=grey(radius+eps);
    return {c-pad,c+pad};
}

// Split the longest centroid extent at the median. Sorting the tiny index
// ranges in place never moves centers (slot identity is used by shading).
inline void build_range(BallBVH &tree,const Balls &balls,int node,uint8_t *slots,int count) {
    if(count==1) {
        tree.tag[node]=int8_t(slots[0]);
        tree.box[node]=bounds(balls.c[slots[0]]);
        return;
    }
    Vec c=balls.c[slots[0]];
    Box centers={c,c};
    for(int i=1;i<count;++i) {
        c=balls.c[slots[i]];
        centers=unite(centers,{c,c});
    }
    Vec d=centers.b-centers.a;
    int ax=d.x>=d.y?0:1;
    if(d.z>axis(d,ax)) ax=2;
    for(int i=1;i<count;++i) {
        uint8_t slot=slots[i];
        float key=axis(balls.c[slot],ax);
        int j=i;
        while(j>0 && axis(balls.c[slots[j-1]],ax)>key) { slots[j]=slots[j-1]; --j; }
        slots[j]=slot;
    }
    // A subtree of k leaves occupies exactly 2*k-1 contiguous nodes.
    int half=count/2,left=node+1,right=left+(2*half-1);
    tree.tag[node]=int8_t(-(node+2*count-1));
    build_range(tree,balls,left,slots,half);
    build_range(tree,balls,right,slots+half,count-half);
    tree.box[node]=unite(tree.box[left],tree.box[right]);
}
} // namespace ball_bvh_detail

inline void build_ball_bvh(BallBVH &tree,const Balls &balls) {
    uint8_t slots[BallBVH::slot_count];
    for(int i=0;i<BallBVH::slot_count;++i) slots[i]=uint8_t(i);
    ball_bvh_detail::build_range(tree,balls,0,slots,BallBVH::slot_count);
}

// Requires a previously built tree. Reverse preorder visits both children
// before their parent, so arbitrary ball motion needs no sort or scratch.
inline void refit_ball_bvh(BallBVH &tree,const Balls &balls) {
    for(int i=BallBVH::node_count-1;i>=0;--i) {
        int tag=tree.tag[i];
        if(tag>=0) tree.box[i]=ball_bvh_detail::bounds(balls.c[tag]);
        else {
            int left=i+1;
            int right=tree.tag[left]<0?-tree.tag[left]:left+1;
            tree.box[i]=unite(tree.box[left],tree.box[right]);
        }
    }
}

// inv=invdir(r.d); r.d is unit length as required by sphere_t. Return true
// only for a ball strictly inside (t0, h.t); leave all of h alone on miss.
ACG_HD inline bool intersect_ball_bvh(const BallBVH &tree,const Balls &balls,Ray r,Vec inv,
                                      float t0,Hit &h) {
    const float limit=h.t;
    float best=limit;
    int slot=BallBVH::slot_count;
    for(int i=0;i<BallBVH::node_count;) {
        int tag=tree.tag[i];
#ifdef ACG_COUNT_TRAVERSAL
        ++ball_counters.boxes;
#endif
        if(!overlaps(tree.box[i],r,inv,t0,best)) { i=tag<0?-tag:i+1; continue; }
        if(tag>=0) {
#ifdef ACG_COUNT_TRAVERSAL
            ++ball_counters.spheres;
#endif
            // sphere_t has a strict upper bound. Use the incoming limit here
            // so a later leaf can still win an exact-distance tie by slot.
            float t;
            if(sphere_t(r,balls.c[tag],radius,t0,limit,t) &&
               (t<best || (t==best && tag<slot))) { best=t; slot=tag; }
        }
        ++i;
    }
    if(slot==BallBVH::slot_count) return false;
    record(h,best,unit((r.o-balls.c[slot])+r.d*best),Ball,slot);
    return true;
}

#endif // ACG_BALL_BVH_H_INCLUDED
