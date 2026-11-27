// Compile from the repository root: g++ -O2 -std=c++17 check_soft_intervals.cpp -o check_soft && ./check_soft
#include "felt_pyramid.h"
#include <cstdio>
#include <initializer_list>
#include <limits>

int main() {
    using namespace acg_scene_detail;
    constexpr int samples=20000;
    const Vec p={0,0,0};
    {
        double sharp=0,filtered=0;
        for(int i=0;i<1024;++i) {
            const float phase=(i+.5f)*pi/1024;
            sharp+=marble_vein(phase,0);
            filtered+=marble_vein(phase,.10f);
        }
        if(fabs(sharp-filtered)/1024>1.e-5 ||
           fabsf(grain({0,0,.33f},.15f)-.53f)>1.e-6f) {
            std::fputs("Texture filtering failed to preserve the unresolved phase mean.\n",stderr);
            return 1;
        }
    }
    {
        // First-hit sequence and interval quadrature must agree with
        // ordinary ray visibility in a genuinely partial ball shadow.
        Balls balls{};
        float positions[32];
        for(auto& value:positions) value=3;
        positions[0]=positions[1]=0;
        place_balls(balls,0,positions);
        const Vec receiver={0,0,.036f},n={0,1,0};
        const Hit hit={1,n,Felt,-1};
        const Material m=varnish({.04f,.25f,.15f},.82f);
        if(!soft_eligible(2,hit,receiver,n,balls.c)) {
            std::fputs("The near-ball fixture cannot use interval integration.\n",stderr);
            return 1;
        }
        const auto view=prepare_brdf_view(m,n,n);
        Vec ray_sum{},interval_sum{},iid_sum{};
        constexpr unsigned count=8192;
        for(unsigned i=0;i<count;++i) {
            Rng a={hash32(i+3127819u)},b=a,c=a;
            ray_sum=ray_sum+lighting(receiver,n,n,m,view,balls,2,a,hit,false,-1.f,false,{i,584013u});
            interval_sum=interval_sum+lighting(receiver,n,n,m,view,balls,2,b,hit,true,-1.f,false,{i,584013u});
            iid_sum=iid_sum+lighting(receiver,n,n,m,view,balls,2,c,hit,false,-1.f,false);
        }
        if(fabsf(ray_sum.y-interval_sum.y)>count*.0025f || fabsf(ray_sum.y-iid_sum.y)>count*.0035f) {
            std::fputs("The first-hit sequence or interval estimator biased a ball penumbra.\n",stderr);
            return 1;
        }
    }
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
    {
        Balls balls{};
        float positions[32];
        for(auto& position:positions) position=9;
        place_balls(balls,0,positions);
        const Vec n={0,1,0}, q={0,light_y,-.9f};
        const Material m=varnish({.4f,.3f,.2f},.6f);
        const Vec delta=q-p;
        const float d2=dot(delta,delta);
        const Vec wi=unit(delta);
        float pdf;
        const BrdfView view=prepare_brdf_view(m,n,n);
        const Vec f=brdf(m,n,n,wi,view,pdf);
        const Vec expected=f*emission()*(wi.y*wi.y*light_area/(d2*area_samples));
        const Vec terminal=direct_sample(p,n,n,p+n*eps,q,emission(),true,m,view,balls,2,1,false);
        const auto close=[](double actual,double reference) { return fabs(actual-reference)<=1e-6; };
        if(!close(terminal.x,expected.x) || !close(terminal.y,expected.y) ||
           !close(terminal.z,expected.z)) {
            std::fputs("Terminal area sample lost unopposed light energy\n",stderr);
            return 1;
        }

        // At unit GGX roughness with wi=n, D=1/pi and G1(wi)=1. The
        // specular term divided by Fresnel and the specular PDF then coincide.
        struct Lobes { float diffuse, f0, probability; };
        const Lobes lobes[]={{.3f,0,0},{.3f,.02f,.16f},{.3f,.2f,.4f},{.01f,.2f,.8f},{0,.2f,1}};
        for(const Lobes lobe:lobes) for(float grazing:{0.f,0x1p-16f}) {
            const Material surface={grey(lobe.diffuse),grey(lobe.f0),1};
            const Vec wo=unit({1,grazing,0});
            float density;
            const Vec value=brdf(surface,n,wo,n,prepare_brdf_view(surface,n,wo),density);
            const double cosine=wo.y;
            const double specular=1/(2*pi*(1+cosine));
            const double fresnel_ref=lobe.f0==0?0:
                lobe.f0+(1-lobe.f0)*pow(1-sqrt((1+cosine)*.5),5);
            const double value_ref=cosine==0?0:
                (1-fresnel_ref)*lobe.diffuse/pi+fresnel_ref*specular;
            const double density_ref=cosine==0?0:(1-lobe.probability)/pi+lobe.probability*specular;
            if(!close(value.x,value_ref) || !close(value.y,value_ref) ||
               !close(value.z,value_ref) || !close(density,density_ref)) {
                std::fputs("Grazing BRDF value or mixture PDF differs from the GGX reference\n",stderr);
                return 1;
            }
        }

        const Vec lit={q.x,light_y-.75f,q.z};
        const Material mixed={grey(.3f),grey(.2f),1};
        const double distance=q.y-lit.y, light_pdf=distance*distance/light_area;
        const double bsdf_pdf=.6/pi+.4/(4*pi);
        const double weighted_pdf=area_samples*light_pdf;
        const double mis=weighted_pdf*weighted_pdf/(weighted_pdf*weighted_pdf+bsdf_pdf*bsdf_pdf);
        const double scale=(.8*.3/pi+.2/(4*pi))*mis/weighted_pdf;
        const Vec competing=direct_sample(lit,n,n,lit+n*eps,q,emission(),true,mixed,
                                          prepare_brdf_view(mixed,n,n),balls,2,-1.f,true);
        if(!close(competing.x,emission().x*scale) || !close(competing.y,emission().y*scale) ||
           !close(competing.z,emission().z*scale)) {
            std::fputs("Uncached area sample differs from the competing light estimate\n",stderr);
            return 1;
        }
    }
    {
        Balls balls{};
        float positions[32];
        for(auto& position:positions) position=3;
        place_balls(balls,0,positions);
        const Vec above_slate={0,eps,0};
        const Vec below_slate={0,-.05f,0};
        const Vec below_emitter={0,light_y-.1f,light_z};
        const Vec above_emitter={0,light_y+.1f,light_z};
        const Vec column_below={0,below_emitter.y,point_light_position.z};
        const Vec column_above={0,above_emitter.y,point_light_position.z};
        const float column_visibility=fabsf(point_light_position.z-light_z)<light_hz?0.f:1.f;
        if(visibility_to_light(above_slate,point_light_position,balls,1)!=1 ||
           visibility_to_light(below_slate,point_light_position,balls,1)!=0 ||
           visibility_to_light(below_emitter,above_emitter,balls,2)!=0 ||
           visibility_to_light(column_below,column_above,balls,2)!=column_visibility) {
            std::fputs("Point shadow changed slate or emitter occlusion.\n",stderr);
            return 1;
        }

        // Only the source lies in the clear column. The ray crosses a
        // partially opaque part of the grille before entering the column.
        const Vec behind_grille={.658f,.12f,-.75f};
        const Vec inside_column={.658f,.69f,-.56f};
#ifdef ACG_OPAQUE_MASK
        constexpr float grille_visibility=0.f;
#else
        constexpr float grille_visibility=.35f;
#endif
        if(!(fabsf(visibility_to_light(behind_grille,inside_column,balls,1)-grille_visibility)<1.e-6f)) {
            std::fputs("Point shadow lost grille attenuation.\n",stderr);
            return 1;
        }

        positions[0]=positions[1]=0;
        place_balls(balls,0,positions);
        const auto ball_cutoffs_match=[&] {
            const float near=2*radius-eps, far=2*radius+3*eps;
            const Vec center={0,radius,0};
            const float inside=nextafterf(near,0), outside=nextafterf(near,1);
            // A dyadic step keeps these short rays vertical with unit direction.
            return visibility_to_light({0,inside,0},{0,inside+0x1p-8f,0},balls,1)==0 &&
                   visibility_to_light({0,outside,0},{0,outside+0x1p-8f,0},balls,1)==1 &&
                   visibility_to_light(center,{0,nextafterf(far,0),0},balls,1)==1 &&
                   visibility_to_light(center,{0,nextafterf(far,1),0},balls,1)==0;
        };
        bool match=ball_cutoffs_match();
#ifdef ACG_PACKED_ROOM
        PackedBalls packed;
        prepare_packed_balls(balls,packed);
        match=ball_cutoffs_match() && match;
#endif
        BallBVH tree;
        build_ball_bvh(tree,balls);
        balls.tree=&tree;
        if(!ball_cutoffs_match() || !match) {
            std::fputs("Point shadow changed the ball hit interval.\n",stderr);
            return 1;
        }
    }
    {
        Balls balls{};
        float positions[32];
        for(auto& position:positions) position=3;
        place_balls(balls,0,positions);

        const Ray overhead={{0,camera_height,0},{0,-1,0}};
        Hit hit;
        if(!intersect(overhead,balls,2,eps,far_t,hit) ||
           hit.mat!=Felt || hit.t!=camera_height || hit.normal.y!=1 ||
           intersect(overhead,balls,2,eps,camera_height,hit) ||
           !intersect(overhead,balls,2,camera_height,far_t,hit) ||
           hit.mat!=Felt || hit.t<=camera_height || hit.normal.y!=-1) {
            std::fputs("Overhead slate changed a hit or an interval endpoint.\n",stderr);
            return 1;
        }
        // Approach the edge of the clear column at nearly the maximum slope.
        const Ray oblique={{table_clear_x-.0201f,camera_height,table_clear_z-.0201f},
                           unit({max_overhead_lateral_speed,-1,max_overhead_lateral_speed})};
        const float oblique_depth=(0.f-oblique.o.y)*recip(oblique.d.y);
        if(!intersect(oblique,balls,2,eps,far_t,hit) || hit.mat!=Felt ||
           hit.t!=oblique_depth || hit.normal.y!=1) {
            std::fputs("Oblique overhead ray missed the slate top.\n",stderr);
            return 1;
        }

        positions[0]=positions[1]=0;
        place_balls(balls,0,positions);
        // The ball and slate both intersect this ray at the ball's exit.
        const Ray tied={{0,radius,0},{0,-1,0}};
        const auto tie_matches=[&] {
            return intersect(tied,balls,2,eps,far_t,hit) && hit.mat==Ball &&
                   hit.id==0 && hit.t==radius && fabsf(hit.normal.y+1)<1.e-6f &&
                   hit.normal.x==0 && hit.normal.z==0 &&
                   !intersect(tied,balls,2,eps,radius,hit);
        };
        bool match=tie_matches();
#ifdef ACG_PACKED_ROOM
        PackedBalls packed;
        prepare_packed_balls(balls,packed);
        match=tie_matches() && match;
#endif
        BallBVH tree;
        build_ball_bvh(tree,balls);
        balls.tree=&tree;
        if(!tie_matches() || !match) {
            std::fputs("Ball and slate tie changed the selected hit.\n",stderr);
            return 1;
        }
    }
    {
        Balls balls{};
        float positions[32];
        for(auto& position:positions) position=3;
        place_balls(balls,0,positions);

        const Vec down={0,-1,0};
        const auto from_above=[&](float x,float z,Hit& hit) {
            return intersect(Ray{{x,camera_height,z},down},balls,2,eps,far_t,hit);
        };
        constexpr float trim_sample=(bore+collar_top_radius)*.5f;
        for(int side:{-1,1}) for(int column:{-1,0,1}) {
            const float x=column*pocket_x,z=side*(column?pocket_z:side_pocket_z);
            Hit catcher,trim;
            if(!from_above(x,z,catcher) || catcher.mat!=Ebony ||
               fabsf(catcher.t-(camera_height-pocket_floor_y))>2*eps ||
               !from_above(x,z+side*(trim_sample+(column?0.f:side_pocket_shear*pocket_top)),trim) ||
               trim.mat!=PocketTrim ||
               fabsf(trim.t-(camera_height-pocket_top))>2*eps || trim.normal.y!=1.f) {
                std::fputs("Pocket center or raised trim gave the wrong surface.\n",stderr);
                return 1;
            }
            if(column && (!from_above(x+column*.067f,z-side*.032f,trim) || trim.mat!=PocketTrim ||
                          fabsf(trim.t-(camera_height-pocket_top))>2*eps || trim.normal.y!=1.f)) {
                std::fputs("Corner trim is missing on the end-rail side.\n",stderr);
                return 1;
            }
        }

        Hit cloth,lining,long_bevel,end_bevel;
        if(!from_above(0,side_pocket_z-trim_sample,cloth) || cloth.mat!=Felt ||
           !from_above((bore+pocket_throat)*.5f,side_pocket_z,lining) ||
           lining.mat!=Leather || lining.normal.x>=0 || lining.normal.y<=0 ||
           !from_above(.4f,.739f+.044f,long_bevel) || long_bevel.mat!=Timber ||
           long_bevel.normal.y<.70f || long_bevel.normal.z<.70f ||
           !from_above(1.374f+.044f,0,end_bevel) || end_bevel.mat!=Timber ||
           end_bevel.normal.y<.70f || end_bevel.normal.x<.70f) {
            std::fputs("Pocket lining, clear cloth, or timber bevel is incorrect.\n",stderr);
            return 1;
        }

        for(int side:{-1,1}) {
            Hit shelf;
            if(!from_above(0,side*(table_nose_z-.015f),shelf) || shelf.mat!=Felt ||
               fabsf(shelf.t-camera_height)>2*eps ||
               visibility_to_light({0,eps,side*.615f},{0,light_y,side*.780f},balls,2)!=1) {
                std::fputs("The side pocket lip overhangs the felt or casts an inboard shadow.\n",stderr);
                return 1;
            }
        }

        for(int side:{-1,1}) for(int column:{-1,0,1}) {
            const float x=column*pocket_x,z=side*(column?pocket_z:side_pocket_z);
            for(float y:{-.010f,-.090f}) {
                const float radial=pocket_opening(y);
                Hit wall;
                if(!from_above(x+radial,z,wall) || wall.mat!=Leather ||
                   fabsf(wall.t-(camera_height-y))>4*eps || wall.normal.x>=0 || wall.normal.y<=0) {
                    std::fputs("An interior pocket taper did not expose its expected depth.\n",stderr);
                    return 1;
                }
            }
        }
        for(int side:{-1,1}) {
            const Ray toward_nose={{.35f,cushion_nose_y,side*.585f},{0,0,float(side)}};
            Hit nose,shoulder;
            if(!intersect(toward_nose,balls,2,eps,far_t,nose) || nose.mat!=CushionCloth ||
               fabsf(nose.t-(table_nose_z-.585f))>2*eps || nose.normal.z!=-float(side) ||
               !from_above(.35f,side*(table_nose_z+.003f),shoulder) ||
               shoulder.mat!=CushionCloth || shoulder.normal.y<=0 || side*shoulder.normal.z>=0 ||
               !(shoulder.t>camera_height-.046f && shoulder.t<camera_height-cushion_nose_y)) {
                std::fputs("Cushion nose lost its contact line or rounded shoulder.\n",stderr);
                return 1;
            }
        }
        for(int side:{-1,1}) for(int end:{-1,1}) {
            const auto cushion_at=[&](float x,float z,Vec d) {
                Hit hit;
                return intersect({{x,cushion_nose_y,z},d},balls,2,eps,.051f,hit) && hit.mat==CushionCloth;
            };
            const float long_tip=table_nose_x-corner_jaw_setback;
            const float end_tip=table_nose_z-corner_jaw_setback;
            const Vec long_dir={0,0,float(side)},end_dir={float(end),0,0};
            if(cushion_at(end*(side_mouth_half-.003f),side*.585f,long_dir) ||
               !cushion_at(end*(side_mouth_half+.003f),side*.585f,long_dir) ||
               cushion_at(end*(long_tip+.003f),side*.585f,long_dir) ||
               !cushion_at(end*(long_tip-.003f),side*.585f,long_dir) ||
               cushion_at(end*1.22f,side*(end_tip+.003f),end_dir) ||
               !cushion_at(end*1.22f,side*(end_tip-.003f),end_dir)) {
                std::fputs("The cushion tips do not define the side and corner pocket mouths.\n",stderr);
                return 1;
            }
        }

        for(float offset:{-.001f,.001f}) {
            Hit joint;
            if(!from_above(rail_x,rail_z-rail_half_width+offset,joint) ||
               joint.mat!=Timber || fabsf(joint.t-(camera_height-rail_top))>2*eps) {
                std::fputs("Timber joint deck is not flush.\n",stderr);
                return 1;
            }
        }

        const Ray rising={{.036f,.046f,.625f},unit({0,.1f,1})};
        const Vec inverse=invdir(rising.d);
        const float cut=(table_nose_z-rising.o.z)*inverse.z;
        Hit face;
        if(!intersect_with_inverse(rising,inverse,balls,2,eps,cut+.001f,face) ||
           face.mat!=PocketTrim || face.t!=cut || face.normal.z!=-1.f ||
           visibility(rising,cut+.001f,balls,2)!=0 ||
           intersect_with_inverse(rising,inverse,balls,2,eps,cut,face) ||
           intersect_with_inverse(rising,inverse,balls,2,cut,cut+.001f,face)) {
            std::fputs("Pocket cut face, shadow, or open ray interval is incorrect.\n",stderr);
            return 1;
        }
    }
    {
        using namespace acg_felt;
        Vec cells[36];
        for(auto& cell:cells) cell={2,3,4};
        const Grid fine={cells,6,6};
        if(coarse_node(fine,1,1).z!=4 || bilinear(fine,9,21,pitch).y!=3 ||
           bilinear(fine,5*pitch,0,pitch).x>=0) {
            std::fputs("Felt pyramid changed constant lighting or extrapolated past the grid.\n",stderr);
            return 1;
        }
        cells[3*6+2]={std::numeric_limits<float>::quiet_NaN(),0,0};
        if(coarse_node(fine,1,1).x>=0 || bilinear(fine,17,17,pitch).x>=0 ||
           bilinear(fine,9,9,pitch).x!=2) {
            std::fputs("Felt pyramid interpolated through a rejected receiver.\n",stderr);
            return 1;
        }
        Vec impulse[13*13],filtered[7*7];
        for(auto& v:impulse) v=grey(1);
        impulse[6*13+6]=grey(2);
        const Grid impulse_grid={impulse,13,13},coarse={filtered,7,7};
        for(int y=0;y<7;++y) for(int x=0;x<7;++x) filtered[y*7+x]=coarse_node(impulse_grid,x,y);
        float excess=0;
        for(int y=0;y<13;++y) for(int x=0;x<13;++x)
            excess+=blend_node(impulse[y*13+x],coarse,x,y).x-1;
        if(fabsf(excess-1)>1.e-5f) {
            std::fputs("Layer blending amplified an interior lighting impulse.\n",stderr);
            return 1;
        }
        if(overhead_slate_clear) {
            Balls balls{};
            float positions[32];
            for(auto& value:positions) value=3;
            place_balls(balls,0,positions);
            Vec p;
            if(anchor(0,0,1920,1080,balls,p) || !anchor(960,540,1920,1080,balls,p)) {
                std::fputs("Felt pyramid admitted a rail or rejected the open slate.\n",stderr);
                return 1;
            }
            const Ray ray={{p.x,camera_height,p.z},{0,-1,0}};
            positions[0]=p.x+.06f; positions[1]=p.z;
            Balls contact{};
            place_balls(contact,0,positions);
            if(anchor(960,540,1920,1080,contact,p)) {
                std::fputs("Felt pyramid admitted a ball contact neighborhood.\n",stderr);
                return 1;
            }
            // Compare linear means at a certified, colocated hit. The same
            // proposal cancels the direct/light-hit partition samplewise;
            // deeper roulette may pick different survivors for each split.
            constexpr int n=8192;
            double sum[3]={},squares[3]={};
            for(int i=0;i<n;++i) {
                Rng a={hash32(i+32957u)},b=a,c=a;
                const FirstLight first={static_cast<uint32_t>(i),4928531u};
                const Vec f=trace(ray,balls,2,0,a,false,-1.f,0.f,first);
                const Vec basis=trace_felt_pass<FeltPass::Basis>(ray,balls,2,0,b,false,-1.f,{},0.f,first);
                const Vec rebuilt=trace_felt_pass<FeltPass::Residual>(ray,balls,2,0,c,false,-1.f,basis,0.f,first);
                const Vec error=rebuilt-f;
                for(int j=0;j<3;++j) {
                    const double v=axis(error,j); sum[j]+=v; squares[j]+=v*v;
                }
            }
            for(int j=0;j<3;++j) if(fabs(sum[j]/n)>5*sqrt(squares[j])/n+2.e-5) {
                std::fputs("Felt basis plus residual changed the colocated radiance mean.\n",stderr);
                return 1;
            }
        }
    }
    std::puts("Felt pyramid support and linear lighting split passed.");
    std::puts("Six pocket openings, raised trim, inner lining and timber bevels passed.");
    std::puts("Overhead slate endpoints and ball priority passed.");
    std::puts("Point shadow ball, slate, grille and emitter occlusion passed.");
    std::puts("72 quadrature slices, blockage, tangency, grazing BRDF and area-light weights passed.");
}
