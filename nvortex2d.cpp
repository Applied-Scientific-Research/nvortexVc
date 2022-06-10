/*
 * nvortexVc - test platform for SIMD-acceleration of an N-vortex solver using Vc
 *
 * Copyright (c) 2017-8 Applied Scientific Research, Inc.
 *   Written by Mark J Stock <markjstock@gmail.com>
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <Vc/Vc>

using Vc::float_v;

static float num_flops_per = 12.f;

// serial (x86) instructions

static inline void nbody_kernel_serial(const float sx,
                                       const float sy,
                                       const float ss,
                                       const float sr,
                                       const float tx,
                                       const float ty,
                                       float* const tax,
                                       float* const tay) {
    // 12 flops
    float dx = sx - tx;
    float dy = sy - ty;
    float r2 = dx*dx + dy*dy + sr*sr;
    r2 = ss/r2;
    *tax += r2 * dy;
    *tay -= r2 * dx;
}

void nbody_serial(const int numSrcs,
                  const float * const sx,
                  const float * const sy,
                  const float * const ss,
                  const float * const sr,
                  const int numTarg,
                  const float * const tx,
                  const float * const ty,
                  float * const tax,
                  float * const tay)
{
    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        tax[i] = 0.0;
        tay[i] = 0.0;
        for (int j = 0; j < numSrcs; j++) {
            nbody_kernel_serial(sx[j], sy[j], ss[j], sr[j],
                                tx[i], ty[i], &tax[i], &tay[i]);
        }
    }
}

// vectorized (Vc) instructions

// 01 - sources are vectorized

static inline void nbody_kernel_Vc_01(const Vc::float_v sx,
                                      const Vc::float_v sy,
                                      const Vc::float_v ss,
                                      const Vc::float_v sr,
                                      const Vc::float_v tx,
                                      const Vc::float_v ty,
                                      Vc::float_v* const tax,
                                      Vc::float_v* const tay) {
    // 12*w flops
    Vc::float_v dx = sx - tx;
    Vc::float_v dy = sy - ty;
    Vc::float_v r2 = dx*dx + dy*dy + sr*sr;
    r2 = ss/r2;
    *tax += r2 * dy;
    *tay -= r2 * dx;
}

void nbody_Vc_01(const int numSrcs,
                 const Vc::float_v * const sx,
                 const Vc::float_v * const sy,
                 const Vc::float_v * const ss,
                 const Vc::float_v * const sr,
                 const int numTarg,
                 const float * const tx,
                 const float * const ty,
                 float * const tax,
                 float * const tay)
{
    // vector versions of the targets
    Vc::float_v vtx, vty, vtax, vtay;

    // scalar over targets
    #pragma omp parallel for private(vtx, vty, vtax, vtay)
    for (int i = 0; i < numTarg; i++) {
        // spread this one target over a vector
        vtx = tx[i];
        vty = ty[i];
        vtax = 0.0f;
        vtay = 0.0f;
        // vectorized over sources
        for (int j = 0; j < numSrcs/Vc::float_v::Size; j++) {
            nbody_kernel_Vc_01(sx[j], sy[j], ss[j], sr[j],
                               vtx, vty, &vtax, &vtay);
        }
        // reduce to scalar
        tax[i] = vtax.sum();
        tay[i] = vtay.sum();
    }
}

// 10 - targets are vectorized

static inline void nbody_kernel_Vc_10(const Vc::float_v sx,
                                      const Vc::float_v sy,
                                      const Vc::float_v ss,
                                      const Vc::float_v sr,
                                      const Vc::float_v tx,
                                      const Vc::float_v ty,
                                      Vc::float_v& tax,
                                      Vc::float_v& tay) {
    // 12*w flops
    const Vc::float_v dx = sx - tx;
    const Vc::float_v dy = sy - ty;
    //const Vc::float_v r2 = ss / (dx*dx + dy*dy + sr*sr);
    Vc::float_v r2 = (dx*dx + dy*dy + sr*sr);
    r2 = ss / r2;
    tax += r2 * dy;
    tay -= r2 * dx;
    //std::cout << "        kernel " << tax << " " << tay << std::endl << std::flush;
}

void nbody_Vc_10(const int numSrcs,
                 const float * const sx,
                 const float * const sy,
                 const float * const ss,
                 const float * const sr,
                 const int numTarg,
                 const Vc::float_v * const tx,
                 const Vc::float_v * const ty,
                 Vc::float_v * const tax,
                 Vc::float_v * const tay)
{
    // vector over targets
    #pragma omp parallel for
    for (int i = 0; i < numTarg/Vc::float_v::Size; i++) {
        // zero the output vectors
        tax[i] = 0.0f;
        tay[i] = 0.0f;
        //std::cout << "    targs " << i << " are at " << vtx << " " << vty << std::endl << std::flush;

        // scalar over sources
        for (int j = 0; j < numSrcs; j++) {
            // spread this one source over a vector
            const Vc::float_v vsx = sx[j];
            const Vc::float_v vsy = sy[j];
            const Vc::float_v vss = ss[j];
            const Vc::float_v vsr = sr[j];

            nbody_kernel_Vc_10(vsx, vsy, vss, vsr,
                               tx[i], ty[i], tax[i], tay[i]);

            //std::cout << "      src " << j << " is at " << sx[j] << " " << sy[j] << std::endl << std::flush;
            //std::cout << "      vels " << vtax << " " << vtay << std::endl << std::flush;
        }
        //std::cout << "      final " << tax[i] << " " << tay[i] << std::endl << std::flush;
    }
}

// 11 - sources AND targets are vectorized

static inline void nbody_kernel_Vc_11(const Vc::float_v sx,
                                      const Vc::float_v sy,
                                      const Vc::float_v ss,
                                      const Vc::float_v sr,
                                      const Vc::float_v tx,
                                      const Vc::float_v ty,
                                      Vc::float_v& tax,
                                      Vc::float_v& tay) {
    // 12*w*w flops
    Vc::float_v dx = sx - tx;
    Vc::float_v dy = sy - ty;
    Vc::float_v r2 = ss / (dx*dx + dy*dy + sr*sr);
    tax += r2 * dy;
    tay -= r2 * dx;
    for (int i = 1; i < Vc::float_v::Size; i++) {
    // shift the sources and do it again
        dx = sx.rotated(i) - tx;
        dy = sy.rotated(i) - ty;
        r2 = ss.rotated(i) / (dx*dx + dy*dy + sr*sr);	// note we assume that radii are constant
        tax += r2 * dy;
        tay -= r2 * dx;
    }
    //std::cout << "        kernel " << tax << " " << tay << std::endl << std::flush;
}

void nbody_Vc_11(const int numSrcs,
                 const Vc::float_v * const sx,
                 const Vc::float_v * const sy,
                 const Vc::float_v * const ss,
                 const Vc::float_v * const sr,
                 const int numTarg,
                 const Vc::float_v * const tx,
                 const Vc::float_v * const ty,
                 Vc::float_v * const tax,
                 Vc::float_v * const tay)
{
    // vector over targets
    #pragma omp parallel for
    for (int i = 0; i < numTarg/Vc::float_v::Size; i++) {
        // zero the output vectors
        tax[i] = 0.0f;
        tay[i] = 0.0f;
        //std::cout << "    targs " << i << " are at " << vtx << " " << vty << std::endl << std::flush;

        // vector over sources
        for (int j = 0; j < numSrcs/Vc::float_v::Size; j++) {

            nbody_kernel_Vc_11(sx[j], sy[j], ss[j], sr[j],
                               tx[i], ty[i], tax[i], tay[i]);

            //std::cout << "      src " << j << " is at " << sx[j] << " " << sy[j] << std::endl << std::flush;
            //std::cout << "      vels " << vtax << " " << vtay << std::endl << std::flush;
        }
        //std::cout << "      final " << tax[i] << " " << tay[i] << std::endl << std::flush;
    }
}


// main program

static void usage() {
    fprintf(stderr, "Usage: nbody [-n=<number>] [simd iterations] [serial iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {

    static unsigned int test_iterations[] = {4, 4, 4, 2};
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
        for (int i = 0; i < 4; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }


    // allocate particle data

    float * sx = new float[numSrcs];
    float * sy = new float[numSrcs];
    float * ss = new float[numSrcs];
    float * sr = new float[numSrcs];
    for (int i = 0; i < numSrcs; i++) {
        sx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sy[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ss[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sr[i] = 1.0 / sqrt((float)numSrcs);
    }

    float * tx = new float[numTargs];
    float * ty = new float[numTargs];
    float * tax = new float[numTargs];
    float * tay = new float[numTargs];
    for (int i = 0; i < numTargs; i++) {
        tx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ty[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        tax[i] = 0.0;
        tay[i] = 0.0;
    }

    Vc::float_v * vsx = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v * vsy = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v * vss = new Vc::float_v[numSrcs/Vc::float_v::Size];
    Vc::float_v * vsr = new Vc::float_v[numSrcs/Vc::float_v::Size];
    for (size_t i = 0; i < numSrcs/Vc::float_v::Size; ++i) {
        size_t idx = Vc::float_v::Size*i;
        for (size_t j = 0; j < Vc::float_v::Size; ++j) {
            vsx[i][j] = sx[idx];
            vsy[i][j] = sy[idx];
            vss[i][j] = ss[idx];
            vsr[i][j] = sr[idx];
            ++idx;
        }
    }

    Vc::float_v * vtx = new Vc::float_v[numTargs/Vc::float_v::Size];
    Vc::float_v * vty = new Vc::float_v[numTargs/Vc::float_v::Size];
    Vc::float_v * vtax = new Vc::float_v[numTargs/Vc::float_v::Size];
    Vc::float_v * vtay = new Vc::float_v[numTargs/Vc::float_v::Size];
    for (size_t i = 0; i < numTargs/Vc::float_v::Size; ++i) {
        size_t idx = Vc::float_v::Size*i;
        for (size_t j = 0; j < Vc::float_v::Size; ++j) {
            vtx[i][j] = tx[idx];
            vty[i][j] = ty[idx];
            vtax[i][j] = tax[idx];
            vtay[i][j] = tay[idx];
            ++idx;
        }
    }


    //
    // Compute the result using the Vc implementation; report the minimum time
    //
    double minVc = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_Vc_01(numSrcs, vsx, vsy, vss, vsr,
                    numTargs, tx, ty, tax, tay);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc = std::min(minVc, elapsed_seconds.count());
    }

    if (test_iterations[0] > 0) {
    printf("[nbody Vc 01]:\t\t[%.6f] seconds\n", minVc);
    printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minVc));

    // Write sample results
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
    printf("\n");
    }


    //
    // Compute the result using the Vc implementation; report the minimum time
    //
    double minVc10 = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_Vc_10(numSrcs, sx, sy, ss, sr,
                    numTargs, vtx, vty, vtax, vtay);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc10 = std::min(minVc10, elapsed_seconds.count());
    }

    if (test_iterations[1] > 0) {
    printf("[nbody Vc 10]:\t\t[%.6f] seconds\n", minVc10);
    printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minVc10));

    // Write sample results
    for (size_t i = 0; i < numTargs/Vc::float_v::Size; ++i) {
        size_t idx = Vc::float_v::Size*i;
        for (size_t j = 0; j < Vc::float_v::Size; ++j) {
            tax[idx] = vtax[i][j];
            tay[idx] = vtay[i][j];
            ++idx;
        }
    }
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
    printf("\n");

    // accumulate minimum
    minVc = std::min(minVc, minVc10);
    }


    //
    // Compute the result using the Vc implementation; report the minimum time
    //
    double minVc11 = 1e30;
    for (unsigned int i = 0; i < test_iterations[2]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_Vc_11(numSrcs, vsx, vsy, vss, vsr,
                    numTargs, vtx, vty, vtax, vtay);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc11 = std::min(minVc11, elapsed_seconds.count());
    }

    if (test_iterations[2] > 0) {
    printf("[nbody Vc 11]:\t\t[%.6f] seconds\n", minVc11);
    printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minVc11));

    // Write sample results
    for (size_t i = 0; i < numTargs/Vc::float_v::Size; ++i) {
        size_t idx = Vc::float_v::Size*i;
        for (size_t j = 0; j < Vc::float_v::Size; ++j) {
            tax[idx] = vtax[i][j];
            tay[idx] = vtay[i][j];
            ++idx;
        }
    }
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
    printf("\n");

    // accumulate minimum
    minVc = std::min(minVc, minVc11);
    }


    //
    // And run the serial implementation a few times, again reporting the minimum
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[3]; ++i) {
        auto start = std::chrono::system_clock::now();
        nbody_serial(numSrcs, sx, sy, ss, sr,
                     numTargs, tx, ty, tax, tay);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("@time of serial run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minSerial = std::min(minSerial, elapsed_seconds.count());
    }

    if (test_iterations[3] > 0) {
    printf("[nbody serial]:\t\t[%.6f] seconds\n", minSerial);
    printf("               \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per/(1.e+9*minSerial));

    // Write sample results
    for (int i = 0; i < 2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
    printf("\n");

    printf("\t\t\t(%.3fx speedup using Vc)\n", minSerial/minVc);
    }

    return 0;
}
