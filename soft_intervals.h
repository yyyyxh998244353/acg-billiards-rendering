#ifndef ACG_SOFT_INTERVALS_H_INCLUDED
#define ACG_SOFT_INTERVALS_H_INCLUDED

// Include inside acg_scene_detail after the room, Hit and lamp declarations.
// The lamp faces -Y; its long axis is X, centred at X=0. All coordinates
// (including slice z and the returned interval endpoints) are world metres.
struct SoftEmitter { float y,z,hx,hz; };
constexpr SoftEmitter soft_lamp={light_y,light_z,light_hx,light_hz};
struct SoftInterval { double a,b; };
struct SoftSlice {
    SoftInterval visible[17]; // sorted, disjoint, positive length; only [0,count) is valid
    int count;
    double geometry;          // integral of cos(receiver)*cos(lamp)/distance^2 dx
};

// Intersect [a,b] with k*x <= c. Only positive-length spans affect radiance.
ACG_HD inline bool soft_clip(double k,double c,double &a,double &b) {
    if(k>0) b=fmin(b,c/k);
    else if(k<0) a=fmax(a,c/k);
    else if(c<0) return false;
    return a<b;
}

// Slice plane coordinates: X is along the emitter, D is along the unit vector
// from the origin towards the line in YZ. The line is at D=length>0. The
// sphere cuts this plane in a disk (u,v), of squared radius r2. Project the
// disk onto the *finite* segments to the emitter, not the double tangent cone.
ACG_HD inline bool soft_disk_interval(double u,double v,double r2,double outside,
                                      double length,double &a,double &b) {
    if(r2<=0) return false; // empty section or point tangency of zero measure
    double r=sqrt(r2);
    if(v+r<=0 || v-r>=length) return false;
    if(outside<0) return true; // origin strictly inside: all segments enter the ball

    // Tangent directions (X,D), ordered by angle measured from forward D.
    // They need not both point forward. Clip with oriented half-planes instead
    // of dividing their X by D (which fails for unbounded shadows). At surface
    // contact t=0 these reduce to the inward half-plane dot(C,ray)>=0.
    double t=sqrt(outside);
    double xl=u*t-v*r, dl=v*t+u*r;
    double xr=u*t+v*r, dr=v*t-u*r;
    if(v+r>length) {
        // A tangent contact beyond the segment end is replaced by the rim
        // at D=length. Contact depth is direction.D * t / |(u,v)|^2.
        double limit=length*(u*u+v*v), dv=length-v;
        double half=sqrt(fmax(0.,(r-dv)*(r+dv)));
        if(dl*t>limit) { xl=u-half; dl=length; }
        if(dr*t>limit) { xr=u+half; dr=length; }
    }
    return soft_clip(-dl,-xl*length,a,b) && soft_clip(dr,xr*length,a,b);
}

// a,b relative to receiver X; h2=height^2, s2=h2+(z-p.z)^2, s=sqrt(s2).
// Integral of h2/(x^2+s2)^2. Use a single atan2 of the angle difference
// rather than subtracting nearly equal antiderivatives for narrow slivers.
ACG_HD inline double soft_cosine_integral(double a,double b,double h2,double s2,double s) {
    double w=b-a, ab=a*b;
    return (.5*h2/s2)*(w*(s2-ab)/((a*a+s2)*(b*b+s2)) + atan2(s*w,s2+ab)/s);
}

// Build one slice at a supplied world z (draw it in [lamp.z-hz,lamp.z+hz]).
// Finite scene inputs, lamp above the receiver, and 16 equal opaque spheres.
// Use shadow_dy=0 for mathematical visibility from p. Use eps after a successful
// soft_eligible check to match direct_sample's displaced visibility origin;
// the two cosines and inverse square ALWAYS use the undisplaced shading point.
// false means unsupported configuration (fall back), NOT a black sample.
ACG_HD inline bool soft_slice(Vec p,const Vec centers[16],double z,SoftSlice &out,
                              SoftEmitter lamp=soft_lamp,double rad=radius,float shadow_dy=0) {
    out.count=0; out.geometry=0;
    const float org_y=p.y+shadow_dy; // same float addition as p+n*eps
    const double h=double(lamp.y)-p.y, dy=double(lamp.y)-org_y, dz=z-p.z;
    if(!(h>0 && dy>0 && lamp.hx>0 && lamp.hz>0 && rad>=0)) return false;
    const double length=sqrt(dy*dy+dz*dz), ey=dy/length, ez=dz/length;
    const double xmin=-double(lamp.hx)-p.x, xmax=double(lamp.hx)-p.x;
    // Use the output array as sorted blocker scratch, in receiver-relative X.
    int nblocked=0;
    for(int i=0;i<16;++i) {
        const double cy=double(centers[i].y)-org_y, cz=double(centers[i].z)-p.z;
        const double perp=cy*ez-cz*ey, r2=(rad-perp)*(rad+perp);
        if(r2<=0) continue;
        const double u=double(centers[i].x)-p.x, v=cy*ey+cz*ez;
        const double outside=u*u+cz*cz+(cy-rad)*(cy+rad);
        double a=xmin,b=xmax;
        if(!soft_disk_interval(u,v,r2,outside,length,a,b)) continue;
        if(a==xmin && b==xmax) return true;
        int j=nblocked++;
        while(j>0 && out.visible[j-1].a>a) { out.visible[j]=out.visible[j-1]; --j; }
        out.visible[j]={a,b};
    }
    // Complement the union in one sweep. We write at most one gap per input
    // blocker, so never overwrite an unread interval (including nested ones).
    double cursor=xmin, h2=h*h, s2=h2+dz*dz, s=shadow_dy==0?length:sqrt(s2);
    for(int i=0;i<=nblocked;++i) {
        const SoftInterval block=i<nblocked?out.visible[i]:SoftInterval{xmax,xmax};
        if(block.a>cursor) {
            out.visible[out.count++]={cursor+p.x,block.a+p.x};
            out.geometry+=soft_cosine_integral(cursor,block.a,h2,s2,s);
        }
        cursor=fmax(cursor,block.b);
    }
    return true;
}

ACG_HD inline bool soft_visible(const SoftSlice &s,double x) {
    for(int i=0;i<s.count;++i) {
        if(x<s.visible[i].a) return false;
        if(x<s.visible[i].b) return true;
    }
    return false;
}

// Receiver-to-rectangle pyramid. Cross-sections are horizontal rectangles;
// each edge coordinate is affine in y. Bound the portion in a box's Y slab
// with the extrema at the two clipped heights. This may reject clear beams,
// but cannot accept one passing through a primitive's bounds. In particular,
// checking just the centre ray or just the receiver's XZ is not sufficient.
struct SoftBeam { Vec org; double height,xlo,xhi,zlo,zhi; };
ACG_HD inline bool soft_beam_overlaps(const SoftBeam &b,Box box) {
    const double pad=.25*eps;
    double y0=fmax(0.,double(box.a.y)-pad-b.org.y);
    double y1=fmin(b.height,double(box.b.y)+pad-b.org.y);
    if(y0>y1) return false;
    return b.org.x+fmin(y0*b.xlo,y1*b.xlo)<=box.b.x+pad &&
           b.org.x+fmax(y0*b.xhi,y1*b.xhi)>=box.a.x-pad &&
           b.org.z+fmin(y0*b.zlo,y1*b.zlo)<=box.b.z+pad &&
           b.org.z+fmax(y0*b.zhi,y1*b.zhi)>=box.a.z-pad;
}

// Certify the whole rectangle once per shading point, before drawing samples.
// Only the front of the flat felt slate qualifies. Query the actual static BVH
// (including cushions, pocket wells, the alpha card and optional glass); no
// assumption that the off-axis lamp is contained by the table footprint.
ACG_HD inline bool soft_eligible(int stage,const Hit &hit,Vec p,Vec n,const Vec centers[16],
                                 SoftEmitter lamp=soft_lamp,double rad=radius) {
    if(stage!=2 || hit.mat!=Felt || hit.normal.y!=1 || n.x!=0 || n.y!=1 || n.z!=0 ||
       fabsf(p.y)>.25f*eps || !(lamp.hx>0 && lamp.hz>0 && rad>=0)) return false;
    Vec org=p+n*eps;
    const double h=double(lamp.y)-org.y;
    if(!(h>4*eps)) return false;
    // The interval solve uses open geometric segments. Do not use it where
    // direct_sample's t_min or shortened t_max might discard a ball hit.
    const double safe=rad+4*eps;
    for(int i=0;i<16;++i) {
        const double x=double(centers[i].x)-org.x, y=double(centers[i].y)-org.y;
        const double z=double(centers[i].z)-org.z;
        if(x*x+y*y+z*z<=safe*safe || y+safe>=h) return false;
    }
    SoftBeam beam={org,h,(-double(lamp.hx)-org.x)/h,(double(lamp.hx)-org.x)/h,
                   (double(lamp.z)-lamp.hz-org.z)/h,(double(lamp.z)+lamp.hz-org.z)/h};
    const StaticScene &r=room();
    for(int i=0;i<r.nodes;) {
        const Node &node=r.n[i];
        if(!soft_beam_overlaps(beam,node.box)) { i=node.escape; continue; }
        for(int j=node.first;j<node.first+node.count;++j)
            if(soft_beam_overlaps(beam,r.p[j].box)) return false;
        ++i;
    }
    return true;
}

// Unbiased integration with scene.h's *existing* MIS partition:
// At an eligible vertex replace ONLY rectangular NEE. For each of N slices
// draw uniform z (stratification OK) and independent uniform x on the lamp.
// Let C be an x-independent RGB control, e.g. m.diffuse/pi, G=h^2/dist^4,
// and V=soft_visible(slice,x). The slice's radiance contribution is
//
//   Le * (2*hz/N) * [ C*slice.geometry
//                   + (2*hx)*V*G*(brdf(wi)*wL - C) ].
//
// wL=power_weight(N*lp, pdf_brdf), lp=dist^2/(wi.y*light_area) in solid
// angle. Skip the residual evaluation if V=0; never clamp negative residuals.
// Conditioning on z, the expectation of the second term subtracts exactly
// the first term's control. What remains is the original full, MIS-weighted
// BRDF integral, including GGX and the direction-dependent diffuse Fresnel
// factor on felt. Do not replace it by albedo times the geometry integral.
// Keep the complementary BSDF-sampled lamp-hit weight (same N and lp),
// point light, indirect bounces and delta hits unchanged. No extra area NEE
// sample on top. Ineligible vertices use the original shadow-ray estimator.

#endif
