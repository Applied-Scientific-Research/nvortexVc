/*
  Copyright (c) 2015,7, Mark J Stock
*/

#include <Vc/Vc>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

using Vc::float_v;
using VectorF = std::vector<float, Vc::Allocator<float>>;

static float num_flops_per = 30.f;

// serial (x86) instructions

static inline void nbody_kernel_serial(float const sx, const float sy, const float sz,
                                       const float ssx,const float ssy,const float ssz, const float sr,
                                       const float tx, const float ty, const float tz, const float tr,
                                       float* tax, float* tay, float* taz) {
    // 30 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    float r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr*tr;
    r2 = 1.0/(r2*sqrt(r2));
    (*tax) += r2 * (dz*ssy - dy*ssz);
    (*tay) += r2 * (dx*ssz - dz*ssx);
    (*taz) += r2 * (dy*ssx - dx*ssy);
}

void nbody_serial(const int numSrcs, const float sx[], const float sy[], const float sz[],
                                     const float ssx[],const float ssy[],const float ssz[], const float sr[],
                  const int numTarg, const float tx[], const float ty[], const float tz[], const float tr[],
                                     float tax[], float tay[], float taz[]) {

    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        tax[i] = 0.0;
        tay[i] = 0.0;
        taz[i] = 0.0;
        for (int j = 0; j < numSrcs; j++) {
            nbody_kernel_serial(sx[j], sy[j], sz[j], ssx[j], ssy[j], ssz[j], sr[j],
                                tx[i], ty[i], tz[i], tr[i], &tax[i], &tay[i], &taz[i]);
        }
    }
}


// vectorized (Vc) instructions

// 01 - sources are vectorized

static inline void nbody_kernel_Vc_01(const Vc::float_v sx, const Vc::float_v sy, const Vc::float_v sz,
                                      const Vc::float_v ssx, const Vc::float_v ssy, const Vc::float_v ssz,
                                      const Vc::float_v sr,
                                      const Vc::float_v tx, const Vc::float_v ty, const Vc::float_v tz,
                                      const Vc::float_v tr,
                                      Vc::float_v* tax, Vc::float_v* tay, Vc::float_v* taz) {
    // 30*w flops
    const Vc::float_v dx = sx - tx;
    const Vc::float_v dy = sy - ty;
    const Vc::float_v dz = sz - tz;
    Vc::float_v r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr*tr;
    //r2 = 1.0 / (r2*sqrt(r2));
    r2 = Vc::reciprocal(r2*Vc::sqrt(r2));
    (*tax) += r2 * (dz*ssy - dy*ssz);
    (*tay) += r2 * (dx*ssz - dz*ssx);
    (*taz) += r2 * (dy*ssx - dx*ssy);
}

// compute directly from the array of Vc::float_v objects
void nbody_Vc_01(const int numSrcs, const Vc::float_v sx[], const Vc::float_v sy[], const Vc::float_v sz[],
                                    const Vc::float_v ssx[],const Vc::float_v ssy[],const Vc::float_v ssz[],
                                    const Vc::float_v sr[],
                 const int numTarg, const float tx[], const float ty[], const float tz[],
                                    const float tr[],
                                    float tax[], float tay[], float taz[]) {

    // scalar over targets
    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        // spread this one target over a vector
        const Vc::float_v vtx = tx[i];
        const Vc::float_v vty = ty[i];
        const Vc::float_v vtz = tz[i];
        const Vc::float_v vtr = tr[i];
        Vc::float_v vtax(0.0f);
        Vc::float_v vtay(0.0f);
        Vc::float_v vtaz(0.0f);
        // vectorized over sources
        for (int j = 0; j < numSrcs/Vc::float_v::Size; j++) {
            nbody_kernel_Vc_01(sx[j], sy[j], sz[j], ssx[j], ssy[j], ssz[j], sr[j],
                               vtx, vty, vtz, vtr, &vtax, &vtay, &vtaz);
        }
        // reduce to scalar
        tax[i] = vtax.sum();
        tay[i] = vtay.sum();
        taz[i] = vtaz.sum();
    }
}

// use simdize to read the std::vector as Vc::float_v objects
void nbody_Vc_02(const int numSrcs, const VectorF& sx, const VectorF& sy, const VectorF& sz,
                                    const VectorF& ssx,const VectorF& ssy,const VectorF& ssz,
                                    const VectorF& sr,
                 const int numTarg, const float tx[], const float ty[], const float tz[],
                                    const float tr[],
                                    float tax[], float tay[], float taz[]) {

    Vc::simdize<VectorF::const_iterator> sxit;
    Vc::simdize<VectorF::const_iterator> syit;
    Vc::simdize<VectorF::const_iterator> szit;
    Vc::simdize<VectorF::const_iterator> ssxit;
    Vc::simdize<VectorF::const_iterator> ssyit;
    Vc::simdize<VectorF::const_iterator> sszit;
    Vc::simdize<VectorF::const_iterator> srit;

    // scalar over targets
    #pragma omp parallel for private(sxit, syit, szit, ssxit, ssyit, sszit, srit)
    for (int i = 0; i < numTarg; i++) {
        // spread this one target over a vector
        const Vc::float_v vtx = tx[i];
        const Vc::float_v vty = ty[i];
        const Vc::float_v vtz = tz[i];
        const Vc::float_v vtr = tr[i];
        Vc::float_v vtax(0.0f);
        Vc::float_v vtay(0.0f);
        Vc::float_v vtaz(0.0f);
        // and convert the source data from std::vector to Vc::float_v
        sxit = sx.begin();
        syit = sy.begin();
        szit = sz.begin();
        ssxit = ssx.begin();
        ssyit = ssy.begin();
        sszit = ssz.begin();
        srit = sr.begin();
        // vectorized over sources
        for (int j = 0; j < numSrcs/Vc::float_v::Size; j++) {
            //if (i==0 and j<100) printf("    j %d has *sxit %ld\n",j,&(*sxit));
            nbody_kernel_Vc_01(*sxit, *syit, *szit, *ssxit, *ssyit, *sszit, *srit,
                               vtx, vty, vtz, vtr, &vtax, &vtay, &vtaz);
            // advance all of the source iterators
            ++sxit;
            ++syit;
            ++szit;
            ++ssxit;
            ++ssyit;
            ++sszit;
            ++srit;
        }
        // reduce to scalar
        tax[i] = vtax.sum();
        tay[i] = vtay.sum();
        taz[i] = vtaz.sum();
    }
}

// convert a std::vector into native Vc array
inline const Vc::Memory<Vc::float_v> stdvec_to_floatvvec (const VectorF& in, const float defaultval) {
    Vc::Memory<Vc::float_v> out(in.size());
    for (size_t i=0; i<in.size(); ++i) out[i] = in[i];
    for (size_t i=in.size(); i<out.entriesCount(); ++i) out[i] = defaultval;
    return out;
}

// fully convert all std::vector to temporary Vc::float_v containers
void nbody_Vc_03(const int numSrcs, const VectorF& sx, const VectorF& sy, const VectorF& sz,
                                    const VectorF& ssx,const VectorF& ssy,const VectorF& ssz,
                                    const VectorF& sr,
                 const int numTarg, const float tx[], const float ty[], const float tz[],
                                    const float tr[],
                                    float tax[], float tay[], float taz[]) {

    // temporary arrays of float_v objects
    const Vc::Memory<Vc::float_v> sxfv = stdvec_to_floatvvec(sx, 0.0);
    const Vc::Memory<Vc::float_v> syfv = stdvec_to_floatvvec(sy, 0.0);
    const Vc::Memory<Vc::float_v> szfv = stdvec_to_floatvvec(sz, 0.0);
    const Vc::Memory<Vc::float_v> ssxfv = stdvec_to_floatvvec(ssx, 0.0);
    const Vc::Memory<Vc::float_v> ssyfv = stdvec_to_floatvvec(ssy, 0.0);
    const Vc::Memory<Vc::float_v> sszfv = stdvec_to_floatvvec(ssz, 0.0);
    const Vc::Memory<Vc::float_v> srfv = stdvec_to_floatvvec(sr, 1.0);

    // scalar over targets
    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        // spread this one target over a vector
        const Vc::float_v vtx = tx[i];
        const Vc::float_v vty = ty[i];
        const Vc::float_v vtz = tz[i];
        const Vc::float_v vtr = tr[i];
        Vc::float_v vtax(0.0f);
        Vc::float_v vtay(0.0f);
        Vc::float_v vtaz(0.0f);
        // vectorized over sources
        for (int j = 0; j < sxfv.vectorsCount(); j++) {
            nbody_kernel_Vc_01(sxfv.vector(j), syfv.vector(j), szfv.vector(j),
                               ssxfv.vector(j), ssyfv.vector(j), sszfv.vector(j), srfv.vector(j),
                               vtx, vty, vtz, vtr, &vtax, &vtay, &vtaz);
        }
        // reduce to scalar
        tax[i] = vtax.sum();
        tay[i] = vtay.sum();
        taz[i] = vtaz.sum();
    }
}


// main program

static void usage() {
    fprintf(stderr, "Usage: nbody [-n=<number>] [simd iterations] [serial iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {

    static unsigned int test_iterations[] = {4, 2};
    const int maxGangSize = 16;
    int numSrcs = maxGangSize*(10000/maxGangSize);
    int numTargs = maxGangSize*(10000/maxGangSize);

    if (argc > 1) {
        if (strncmp(argv[1], "-n=", 3) == 0) {
            int num = atof(argv[1] + 3);
            if (num < 1) usage();
            numSrcs = maxGangSize*(num/maxGangSize);
            numTargs = maxGangSize*(num/maxGangSize);
        }
    }
    if ((argc == 3) || (argc == 4)) {
        for (int i = 0; i < 2; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }


    // allocate particle data

    float *sx = new float[numSrcs];
    float *sy = new float[numSrcs];
    float *sz = new float[numSrcs];
    float *ssx = new float[numSrcs];
    float *ssy = new float[numSrcs];
    float *ssz = new float[numSrcs];
    float *sr = new float[numSrcs];
    for (int i = 0; i < numSrcs; i++) {
        sx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sy[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ssx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ssy[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ssz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sr[i] = 1.0 / sqrt((float)numSrcs);
    }

    float *tx = new float[numTargs];
    float *ty = new float[numTargs];
    float *tz = new float[numTargs];
    float *tr = new float[numTargs];
    float *tax = new float[numTargs];
    float *tay = new float[numTargs];
    float *taz = new float[numTargs];
    for (int i = 0; i < numTargs; i++) {
        tx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ty[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        tz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        tr[i] = 1.0 / sqrt((float)numTargs);
        tax[i] = 0.0;
        tay[i] = 0.0;
        taz[i] = 0.0;
    }

    // vectorize over arrays of float_v types
    Vc::float_v *vsx = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v *vsy = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v *vsz = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v *vssx = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v *vssy = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v *vssz = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v *vsr = new Vc::float_v[numSrcs/Vc::float_v::Size];
    for (size_t i = 0; i < numSrcs/Vc::float_v::Size; ++i) {
        size_t idx = Vc::float_v::Size*i;
        for (size_t j = 0; j < Vc::float_v::Size; ++j) {
            vsx[i][j] = sx[idx];
            vsy[i][j] = sy[idx];
            vsz[i][j] = sz[idx];
            vssx[i][j] = ssx[idx];
            vssy[i][j] = ssy[idx];
            vssz[i][j] = ssz[idx];
            vsr[i][j] = sr[idx];
            ++idx;
        }
    }
    printf("vsx is %d * %d\n",numSrcs/Vc::float_v::Size,sizeof(Vc::float_v));

    // vectorize over standard library vector objects
    VectorF svsx;
    VectorF svsy;
    VectorF svsz;
    VectorF svssx;
    VectorF svssy;
    VectorF svssz;
    VectorF svsr;
    svsx.resize(numSrcs);
    svsy.resize(numSrcs);
    svsz.resize(numSrcs);
    svssx.resize(numSrcs);
    svssy.resize(numSrcs);
    svssz.resize(numSrcs);
    svsr.resize(numSrcs);
    for (size_t i = 0; i < numSrcs; ++i) {
        svsx[i] = sx[i];
        svsy[i] = sy[i];
        svsz[i] = sz[i];
        svssx[i] = ssx[i];
        svssy[i] = ssy[i];
        svssz[i] = ssz[i];
        svsr[i] = sr[i];
    }
    printf("svsx is %d * %d\n",svsx.size(),sizeof(float));
    printf("svsx is at %ld, which off alignment by %ld\n",svsx.data(),(size_t)(svsx.data())%32);
    printf("svsy is at %ld, which off alignment by %ld\n",svsy.data(),(size_t)(svsy.data())%32);
    printf("svsz is at %ld, which off alignment by %ld\n",svsz.data(),(size_t)(svsz.data())%32);


    //
    // Compute the result using the Vc implementation; report the minimum time
    //
    double minVc = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_Vc_01(numSrcs, vsx, vsy, vsz, vssx, vssy, vssz, vsr,
                    numTargs, tx, ty, tz, tr, tax, tay, taz);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc = std::min(minVc, elapsed_seconds.count());
    }

    printf("[nbody Vc 01]:\t\t[%.6f] seconds\n", minVc);
    printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minVc));

    // Write sample results
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g %g\n",i,tax[i],tay[i],taz[i]);
    printf("\n");


    //
    // Compute the result using Vc over std::vector objects with simdize; report the minimum time
    //
    minVc = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_Vc_02(numSrcs, svsx, svsy, svsz, svssx, svssy, svssz, svsr,
                    numTargs, tx, ty, tz, tr, tax, tay, taz);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc = std::min(minVc, elapsed_seconds.count());
    }

    printf("[nbody Vc 02]:\t\t[%.6f] seconds\n", minVc);
    printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minVc));

    // Write sample results
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g %g\n",i,tax[i],tay[i],taz[i]);
    printf("\n");


    //
    // Compute the result using Vc over std::vector objects with copying; report the minimum time
    //
    minVc = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_Vc_03(numSrcs, svsx, svsy, svsz, svssx, svssy, svssz, svsr,
                    numTargs, tx, ty, tz, tr, tax, tay, taz);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc = std::min(minVc, elapsed_seconds.count());
    }

    printf("[nbody Vc 03]:\t\t[%.6f] seconds\n", minVc);
    printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minVc));

    // Write sample results
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g %g\n",i,tax[i],tay[i],taz[i]);
    printf("\n");


    //
    // And run the serial implementation a few times, again reporting the minimum
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_serial(numSrcs, sx, sy, sz, ssx, ssy, ssz, sr,
                     numTargs, tx, ty, tz, tr, tax, tay, taz);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of serial run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minSerial = std::min(minSerial, elapsed_seconds.count());
    }

    printf("[nbody serial]:\t\t[%.6f] seconds\n", minSerial);
    printf("               \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minSerial));

    // Write sample results
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g %g\n",i,tax[i],tay[i],taz[i]);
    printf("\n");

    printf("\t\t\t(%.3fx speedup using Vc)\n", minSerial/minVc);

    return 0;
}
