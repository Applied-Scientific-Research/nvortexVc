/*
 * nvortexVc - test platform for SIMD-acceleration of an N-vortex solver using Vc
 *
 * Copyright (c) 2017-9,22 Applied Scientific Research, Inc.
 *   Mark J Stock <markjstock@gmail.com>
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_VC
#include <Vc/Vc>

using Vc::float_v;
using VectorF = std::vector<float, Vc::Allocator<float>>;
#endif

static float num_flops_per = 12.f;

// serial (x86) instructions

static inline void nbody_kernel_serial(const float sx, const float sy,
                                       const float ss, const float sr,
                                       const float tx, const float ty,
                                       float* const __restrict__ tax, float* const __restrict__ tay) {
    // 12 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    float r2 = dx*dx + dy*dy + sr*sr;
    r2 = ss/r2;
    *tax += r2 * dy;
    *tay -= r2 * dx;
}

void nbody_serial(const int numSrcs,
                  const float* const __restrict__ sx,
                  const float* const __restrict__ sy,
                  const float* const __restrict__ ss,
                  const float* const __restrict__ sr,
                  const int numTarg,
                  const float* const __restrict__ tx,
                  const float* const __restrict__ ty,
                  float* const __restrict__ tax,
                  float* const __restrict__ tay)
{
    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        for (int j = 0; j < numSrcs; j++) {
            nbody_kernel_serial(sx[j], sy[j], ss[j], sr[j],
                                tx[i], ty[i], &tax[i], &tay[i]);
        }
    }
}


// vectorized (Vc) instructions

// 01 - sources are vectorized

#ifdef USE_VC
static inline void nbody_kernel_Vc_01(const Vc::float_v sx, const Vc::float_v sy,
                                      const Vc::float_v ss,
                                      const Vc::float_v sr,
                                      const Vc::float_v tx, const Vc::float_v ty,
                                      Vc::float_v* const tax,
                                      Vc::float_v* const tay) {
    // 12*w flops
    const Vc::float_v dx = sx - tx;
    const Vc::float_v dy = sy - ty;
    Vc::float_v r2 = dx*dx + dy*dy + sr*sr;
    r2 = ss/r2;
    *tax += r2 * dy;
    *tay -= r2 * dx;
}

// compute directly from the array of Vc::float_v objects
void nbody_Vc_01(const int numSrcs, const Vc::float_v* const sx, const Vc::float_v* const sy,
                                    const Vc::float_v* const ss, const Vc::float_v* const sr,
                 const int numTarg, const float* const tx, const float* const ty,
                                    float* const tax, float* const tay) {

    const int nSrcVec = (numSrcs + Vc::float_v::Size - 1) / Vc::float_v::Size;

    // scalar over targets
    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        // spread this one target over a vector
        const Vc::float_v vtx = tx[i];
        const Vc::float_v vty = ty[i];
        Vc::float_v vtax(0.0f);
        Vc::float_v vtay(0.0f);
        // vectorized over sources
        for (int j = 0; j < nSrcVec; j++) {
            nbody_kernel_Vc_01(sx[j], sy[j], ss[j], sr[j],
                               vtx, vty, &vtax, &vtay);
        }
        // reduce to scalar
        tax[i] += vtax.sum();
        tay[i] += vtay.sum();
    }
}

// use simdize to read the std::vector as Vc::float_v objects

void nbody_Vc_02(const int numSrcs, const float* const sx, const float* const sy,
                                    const float* const ss, const float* const sr,
                 const int numTarg, const Vc::float_v* const tx, const Vc::float_v* const ty,
                                    Vc::float_v* const tax, Vc::float_v* const tay)
{
    const int nTrgVec = (numTarg + Vc::float_v::Size - 1) / Vc::float_v::Size;

    // vector over targets
    #pragma omp parallel for
    for (int i = 0; i < nTrgVec; i++) {
        //std::cout << "    targs " << i << " are at " << vtx << " " << vty << std::endl << std::flush;

        // scalar over sources
        for (int j = 0; j < numSrcs; j++) {
            // spread this one source over a vector
            const Vc::float_v vsx = sx[j];
            const Vc::float_v vsy = sy[j];
            const Vc::float_v vss = ss[j];
            const Vc::float_v vsr = sr[j];

            nbody_kernel_Vc_01(vsx, vsy, vss, vsr,
                               tx[i], ty[i], &tax[i], &tay[i]);

            //std::cout << "      src " << j << " is at " << sx[j] << " " << sy[j] << std::endl << std::flush;
            //std::cout << "      vels " << vtax << " " << vtay << std::endl << std::flush;
        }
        //std::cout << "      final " << tax[i] << " " << tay[i] << std::endl << std::flush;
    }
}

// 03 - sources AND targets are vectorized

static inline void nbody_kernel_Vc_03(const Vc::float_v sx,
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
    // WHY IS THIS 1?
    for (int i = 1; i < (int)Vc::float_v::Size; i++) {
    // shift the sources and do it again
        dx = sx.rotated(i) - tx;
        dy = sy.rotated(i) - ty;
        r2 = ss.rotated(i) / (dx*dx + dy*dy + sr*sr);	// note we assume that radii are constant
        tax += r2 * dy;
        tay -= r2 * dx;
    }
    //std::cout << "        kernel " << tax << " " << tay << std::endl << std::flush;
}

void nbody_Vc_03(const int numSrcs,
                 const Vc::float_v* const sx,
                 const Vc::float_v* const sy,
                 const Vc::float_v* const ss,
                 const Vc::float_v* const sr,
                 const int numTarg,
                 const Vc::float_v* const tx,
                 const Vc::float_v* const ty,
                 Vc::float_v* const tax,
                 Vc::float_v* const tay)
{
    const int nTrgVec = (numTarg + Vc::float_v::Size - 1) / Vc::float_v::Size;
    const int nSrcVec = (numSrcs + Vc::float_v::Size - 1) / Vc::float_v::Size;

    // vector over targets
    #pragma omp parallel for
    for (int i = 0; i < nTrgVec; i++) {
        //std::cout << "    targs " << i << " are at " << vtx << " " << vty << std::endl << std::flush;

        // vector over sources
        for (int j = 0; j < nSrcVec; j++) {

            nbody_kernel_Vc_03(sx[j], sy[j], ss[j], sr[j],
                               tx[i], ty[i], tax[i], tay[i]);

            //std::cout << "      src " << j << " is at " << sx[j] << " " << sy[j] << std::endl << std::flush;
            //std::cout << "      vels " << vtax << " " << vtay << std::endl << std::flush;
        }
        //std::cout << "      final " << tax[i] << " " << tay[i] << std::endl << std::flush;
    }
}

// zero a float_v array
void zero_floatvarry(Vc::float_v* const in, const int n, const float defaultval) {
    const Vc::float_v zvec = Vc::float_v(defaultval);
    size_t nvec = (n + Vc::float_v::Size - 1) / Vc::float_v::Size;
    for (size_t i = 0; i < nvec; ++i) in[i] = zvec;
}

// convert a C-style float array into a C-style float_v array
inline Vc::float_v* floatarry_to_floatvarry (const float* const in, const int n, const float defaultval) {
    size_t nvec = (n + Vc::float_v::Size - 1) / Vc::float_v::Size;
    Vc::float_v* out = new Vc::float_v[nvec];

    for (size_t i = 0; i < nvec-1; ++i) {
        size_t idx = i * Vc::float_v::Size;
        for (size_t j = 0; j < Vc::float_v::Size; ++j) {
            out[i][j] = in[j+idx];
        }
    }

    // last vector may need some default values
    size_t lastj = n - (nvec-1)*Vc::float_v::Size;
    for (size_t j = 0; j < lastj; ++j) {
        size_t idx = (nvec-1) * Vc::float_v::Size;
        out[nvec-1][j] = in[j+idx];
    }
    for (size_t j = lastj; j < Vc::float_v::Size; ++j) {
        out[nvec-1][j] = defaultval;
    }

    return out;
}
#endif

// not really alignment, just minimum block sizes
int buffer(const int _n, const int _align) {
  // 63,64 returns 1*64; 64,64 returns 1*64; 65,64 returns 2*64=128
  return _align*((_n+_align-1)/_align);
}

// distribute ntotal items across nproc processes, now how many are on proc iproc?
int nthisproc(const int ntotal, const int iproc, const int nproc) {
  const int base_nper = ntotal / nproc;
  return (iproc < (ntotal-base_nper*nproc)) ? base_nper+1 : base_nper;
}

// simple struct to hold sources
struct Sources {

  int n = 0;
  float* x = nullptr;
  float* y = nullptr;
  float* r = nullptr;
  float* ss = nullptr;

  // sized constructor
  Sources(const int _n) : n(_n) {
    x = new float[_n];
    y = new float[_n];
    r = new float[_n];
    ss = new float[_n];
  }

  // default constructor - keep pointers null
  Sources() {}

  // copy constructor
  Sources(const Sources& _s) : Sources(_s.n) {
    deep_copy(_s);
  }

  void init_rand(std::mt19937 _gen) {
    std::uniform_real_distribution<> zmean_dist(-1.0, 1.0);
    for (int i=0; i<n; ++i) x[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) y[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) ss[i] = zmean_dist(_gen);
    const float rad = 1.0 / std::sqrt((float)n);
    for (int i=0; i<n; ++i) r[i] = rad;
  }

  void deep_copy(const Sources& _s) {
    assert(n == _s.n && "Copying from Sources with different n");
    for (int i=0; i<n; ++i) x[i] = _s.x[i];
    for (int i=0; i<n; ++i) y[i] = _s.y[i];
    for (int i=0; i<n; ++i) r[i] = _s.r[i];
    for (int i=0; i<n; ++i) ss[i] = _s.ss[i];
  }

  void shallow_copy(const Sources& _s) {
    // warning: this can orphan memory!!!
    n = _s.n;
    x = _s.x;
    y = _s.y;
    r = _s.r;
    ss = _s.ss;
  }

  // destructor
  ~Sources() {
    //if (x != nullptr) delete x;
    //if (y != nullptr) delete y;
    //if (r != nullptr) delete r;
    //if (ss != nullptr) delete ss;
  }
};

#ifdef USE_MPI
// non-class method to swap sources
void exchange_sources(const Sources& sendsrcs, const int ito,
                            Sources& recvsrcs, const int ifrom,
                            MPI_Request* handle) {

  // how many to send and receive?
  MPI_Sendrecv(&sendsrcs.n, 1, MPI_INT, ito, 9, &recvsrcs.n, 1, MPI_INT, ifrom, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //std::cout << "  proc " << iproc << " sending " << sendsrcs.n << " and receiving " << recvsrcs.n << std::endl;

  // do the transfers, saving the handle for future reference
  MPI_Irecv(recvsrcs.x, recvsrcs.n, MPI_FLOAT, ifrom, 10, MPI_COMM_WORLD, &handle[0]);
  MPI_Isend(sendsrcs.x, sendsrcs.n, MPI_FLOAT, ito, 10, MPI_COMM_WORLD, &handle[1]);
  MPI_Irecv(recvsrcs.y, recvsrcs.n, MPI_FLOAT, ifrom, 11, MPI_COMM_WORLD, &handle[2]);
  MPI_Isend(sendsrcs.y, sendsrcs.n, MPI_FLOAT, ito, 11, MPI_COMM_WORLD, &handle[3]);
  MPI_Irecv(recvsrcs.r, recvsrcs.n, MPI_FLOAT, ifrom, 12, MPI_COMM_WORLD, &handle[4]);
  MPI_Isend(sendsrcs.r, sendsrcs.n, MPI_FLOAT, ito, 12, MPI_COMM_WORLD, &handle[5]);
  MPI_Irecv(recvsrcs.ss, recvsrcs.n, MPI_FLOAT, ifrom, 13, MPI_COMM_WORLD, &handle[6]);
  MPI_Isend(sendsrcs.ss, sendsrcs.n, MPI_FLOAT, ito, 13, MPI_COMM_WORLD, &handle[7]);
}
#endif

static void usage() {
    fprintf(stderr, "Usage: nvortex2d.bin [-n=<number>] [simd iterations] [serial iterations]\n");
    exit(1);
}

// main program

int main(int argc, char *argv[]) {

    static unsigned int test_iterations[] = {4, 4, 4, 2};
    int n_in = 10000;
    int nproc = 1;
    int iproc = 0;

    if (argc > 1) {
        if (strncmp(argv[1], "-n=", 3) == 0) {
            int num = atof(argv[1] + 3);
            if (num < 1) usage();
            n_in = num;
        }
    }
    if (argc > 2) {
        test_iterations[0] = atoi(argv[2]);
        test_iterations[1] = test_iterations[0];
        test_iterations[2] = test_iterations[0];
    }
    if (argc > 3) {
        test_iterations[3] = atoi(argv[3]);
    }

#ifdef USE_MPI
    (void) MPI_Init(&argc, &argv);
    (void) MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    if (iproc==0) std::cout << "MPI-capable binary" << std::endl;

    // user specifies *total* number of particles, each processor take a share of those
    n_in = nthisproc(n_in, iproc, nproc);

    std::cout << "Proc " << iproc << " of " << nproc << " owns " << n_in << " particles" << std::endl;
#else
    std::cout << "Running with " << n_in << " particles" << std::endl;
#endif

    // set problem size
    const int maxGangSize = 16;		// this is 512 bits / 32 bits per float
    const int numSrcs = buffer(n_in,maxGangSize);
    const int numTargs = numSrcs;

    // init random number generator
    //std::random_device rd;  //Will be used to obtain a seed for the random number engine
    //std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(12345); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> zmean_dist(-1.0, 1.0);

    // allocate original particle data (used for x86 reference calculation)
    Sources src = Sources(numSrcs);
    src.init_rand(gen);

#ifdef USE_MPI
    // what's the largest number that we'll encounter?
    int n_large = 0;
    MPI_Allreduce(&numSrcs, &n_large, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // make two sets of arrays for communication
    Sources work = Sources(n_large);
    Sources buf = Sources(n_large);
    // and one set of pointers
    Sources ptr;

    // always pull from left (-), send to right (+)
    const int ifrom = (iproc == 0) ? nproc-1 : iproc-1;
    const int ito = (iproc == nproc-1) ? 0 : iproc+1;
    const int nreqs = 8;
    MPI_Request* xfers = new MPI_Request[nreqs];
    for (int ix=0; ix<nreqs; ++ix) xfers[ix] = MPI_REQUEST_NULL;
    //std::cout << "  proc " << iproc << " always sending to " << ito << " and receiving from " << ifrom << std::endl;
#endif

    // allocate target particle data
    //Targets trg = Targets(numTargs);
    //trg.init_rand(gen);
    float* tx = new float[numTargs];
    float* ty = new float[numTargs];
    //float* tr = new float[numTargs];
    float* tax = new float[numTargs];
    float* tay = new float[numTargs];
    for (int i=0; i<numTargs; i++) tx[i] = zmean_dist(gen);
    for (int i=0; i<numTargs; i++) ty[i] = zmean_dist(gen);
    //for (int i=0; i<numTargs; i++) tr[i] = 1.0 / std::sqrt((float)numTargs);
    for (int i=0; i<numTargs; i++) tax[i] = 0.0f;
    for (int i=0; i<numTargs; i++) tay[i] = 0.0f;

#ifdef USE_VC
    // vectorize over arrays of float_v types
    Vc::float_v* vtx = floatarry_to_floatvarry(tx, numTargs, 0.0);
    Vc::float_v* vty = floatarry_to_floatvarry(ty, numTargs, 0.0);
    //Vc::float_v* vtr = floatarry_to_floatvarry(tr, numTargs, 1.0);
    Vc::float_v* vtax = floatarry_to_floatvarry(tax, numTargs, 0.0);
    Vc::float_v* vtay = floatarry_to_floatvarry(tay, numTargs, 0.0);


    //
    // Compute the result using the Vc implementation; report the minimum time
    //
    double minVc = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        auto start = std::chrono::system_clock::now();

        // zero the vels
        for (int j=0; j<numTargs; ++j) tax[j] = 0.0;
        for (int j=0; j<numTargs; ++j) tay[j] = 0.0;

#ifdef USE_MPI
        // first set to compute is self
        ptr.shallow_copy(src);

        for (int ibatch=0; ibatch<nproc ; ++ibatch) {
            // post non-blocking sends (from ptr) and receives (into buf)
            if (ibatch < nproc-1) {
                exchange_sources(ptr, ito, buf, ifrom, xfers);
            }

            Vc::float_v* vsx = floatarry_to_floatvarry(ptr.x, ptr.n, 0.0);
            Vc::float_v* vsy = floatarry_to_floatvarry(ptr.y, ptr.n, 0.0);
            Vc::float_v* vss = floatarry_to_floatvarry(ptr.ss, ptr.n, 0.0);
            Vc::float_v* vsr = floatarry_to_floatvarry(ptr.r, ptr.n, 1.0);

            // run the O(N^2) calculation concurrently using ptr for sources
            nbody_Vc_01(ptr.n, vsx, vsy, vss, vsr,
                        numTargs, tx, ty, tax, tay);

            // wait for new data to arrive before continuing
            if (ibatch < nproc-1) {
                // wait on all transfers to complete
                MPI_Waitall(nreqs, xfers, MPI_STATUS_IGNORE);
                // copy buf to work
                work.deep_copy(buf);
                // ptr now points at work
                ptr.shallow_copy(work);
            }
        }

        // and to ensure correct timings, wait for all MPI processes to finish
        MPI_Barrier(MPI_COMM_WORLD);
#else
        // vectorize over arrays of float_v types (used in Vc01)
        Vc::float_v* vsx = floatarry_to_floatvarry(src.x, numSrcs, 0.0);
        Vc::float_v* vsy = floatarry_to_floatvarry(src.y, numSrcs, 0.0);
        Vc::float_v* vss = floatarry_to_floatvarry(src.ss, numSrcs, 0.0);
        Vc::float_v* vsr = floatarry_to_floatvarry(src.r, numSrcs, 1.0);

        nbody_Vc_01(numSrcs, vsx, vsy, vss, vsr,
                    numTargs, tx, ty, tax, tay);
#endif

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        if (iproc==0) printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc = std::min(minVc, elapsed_seconds.count());
    }

    if (test_iterations[0] > 0 and iproc==0) {
        printf("[nbody Vc 01]:\t\t[%.6f] seconds\n", minVc);
        printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per*nproc*nproc/(1.e+9*minVc));

        // Write sample results
        for (int i=0; i<2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
        printf("\n");
    }


    //
    // Compute the result using Vc over std::vector objects with simdize; report the minimum time
    //
    double minVc02 = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        auto start = std::chrono::system_clock::now();

        // zero target vel vectors
        zero_floatvarry(vtax, numTargs, 0.0);
        zero_floatvarry(vtay, numTargs, 0.0);

#ifdef USE_MPI
        // first set to compute is self
        ptr.shallow_copy(src);

        for (int ibatch=0; ibatch<nproc ; ++ibatch) {
            // post non-blocking sends (from ptr) and receives (into buf)
            if (ibatch < nproc-1) {
                exchange_sources(ptr, ito, buf, ifrom, xfers);
            }

            nbody_Vc_02(ptr.n, ptr.x, ptr.y, ptr.ss, ptr.r,
                        numTargs, vtx, vty, vtax, vtay);

            // wait for new data to arrive before continuing
            if (ibatch < nproc-1) {
                // wait on all transfers to complete
                MPI_Waitall(nreqs, xfers, MPI_STATUS_IGNORE);
                // copy buf to work
                work.deep_copy(buf);
                // ptr now points at work
                ptr.shallow_copy(work);
            }
        }

        // and to ensure correct timings, wait for all MPI processes to finish
        MPI_Barrier(MPI_COMM_WORLD);
#else
        nbody_Vc_02(numSrcs, src.x, src.y, src.ss, src.r,
                    numTargs, vtx, vty, vtax, vtay);
#endif

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        if (iproc==0) printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc02 = std::min(minVc02, elapsed_seconds.count());
    }

    if (test_iterations[1] > 0 and iproc==0) {
        printf("[nbody Vc 02]:\t\t[%.6f] seconds\n", minVc02);
        printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per*nproc*nproc/(1.e+9*minVc02));

        // convert out of float_v
        const size_t nvec = (numTargs + Vc::float_v::Size - 1) / Vc::float_v::Size;
        for (size_t k = 0; k < nvec; ++k) {
            size_t idx = Vc::float_v::Size*k;
            for (size_t j = 0; j < Vc::float_v::Size; ++j) {
                tax[idx] = vtax[k][j];
                tay[idx] = vtay[k][j];
                ++idx;
            }
        }

        // Write sample results
        for (int i=0; i<2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
        printf("\n");

        // accumulate minimum
        minVc = std::min(minVc, minVc02);
    }


    //
    // Compute the result using Vc over std::vector objects with copying; report the minimum time
    //
    double minVc03 = 1e30;
    for (unsigned int i = 0; i < test_iterations[2]; ++i) {
        auto start = std::chrono::system_clock::now();

        // zero target vel vectors
        zero_floatvarry(vtax, numTargs, 0.0);
        zero_floatvarry(vtay, numTargs, 0.0);

#ifdef USE_MPI
        // first set to compute is self
        ptr.shallow_copy(src);

        for (int ibatch=0; ibatch<nproc ; ++ibatch) {
            // post non-blocking sends (from ptr) and receives (into buf)
            if (ibatch < nproc-1) {
                exchange_sources(ptr, ito, buf, ifrom, xfers);
            }

            // vectorize over arrays of float_v types
            Vc::float_v* vsx = floatarry_to_floatvarry(ptr.x, ptr.n, 0.0);
            Vc::float_v* vsy = floatarry_to_floatvarry(ptr.y, ptr.n, 0.0);
            Vc::float_v* vss = floatarry_to_floatvarry(ptr.ss, ptr.n, 0.0);
            Vc::float_v* vsr = floatarry_to_floatvarry(ptr.r, ptr.n, 1.0);

            nbody_Vc_03(ptr.n, vsx, vsy, vss, vsr,
                        numTargs, vtx, vty, vtax, vtay);

            // wait for new data to arrive before continuing
            if (ibatch < nproc-1) {
                // wait on all transfers to complete
                MPI_Waitall(nreqs, xfers, MPI_STATUS_IGNORE);
                // copy buf to work
                work.deep_copy(buf);
                // ptr now points at work
                ptr.shallow_copy(work);
            }
        }

        // and to ensure correct timings, wait for all MPI processes to finish
        MPI_Barrier(MPI_COMM_WORLD);
#else
        // vectorize over arrays of float_v types
        Vc::float_v* vsx = floatarry_to_floatvarry(src.x, numSrcs, 0.0);
        Vc::float_v* vsy = floatarry_to_floatvarry(src.y, numSrcs, 0.0);
        Vc::float_v* vss = floatarry_to_floatvarry(src.ss, numSrcs, 0.0);
        Vc::float_v* vsr = floatarry_to_floatvarry(src.r, numSrcs, 1.0);

        nbody_Vc_03(numSrcs, vsx, vsy, vss, vsr,
                    numTargs, vtx, vty, vtax, vtay);
#endif

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        if (iproc==0) printf("@time of Vc run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minVc03 = std::min(minVc03, elapsed_seconds.count());
    }

    if (test_iterations[2] > 0 and iproc==0) {
        printf("[nbody Vc 03]:\t\t[%.6f] seconds\n", minVc03);
        printf("              \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per*nproc*nproc/(1.e+9*minVc03));

        // convert out of float_v
        const size_t nvec = (numTargs + Vc::float_v::Size - 1) / Vc::float_v::Size;
        for (size_t k = 0; k < nvec; ++k) {
            size_t idx = Vc::float_v::Size*k;
            for (size_t j = 0; j < Vc::float_v::Size; ++j) {
                tax[idx] = vtax[k][j];
                tay[idx] = vtay[k][j];
                ++idx;
            }
        }

        // Write sample results
        for (int i=0; i<2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
        printf("\n");

        // accumulate minimum
        minVc = std::min(minVc, minVc03);
    }

    // save results for error estimate
    std::vector<float> tax_vec(tax, tax+numTargs);
#endif	// USE_VC


    //
    // And run the serial implementation a few times, again reporting the minimum
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[3]; ++i) {
        auto start = std::chrono::system_clock::now();

        // zero the vels
        for (int j=0; j<numTargs; ++j) tax[j] = 0.0;
        for (int j=0; j<numTargs; ++j) tay[j] = 0.0;

#ifdef USE_MPI
        // first set to compute is self
        ptr.shallow_copy(src);

        for (int ibatch=0; ibatch<nproc ; ++ibatch) {
            // post non-blocking sends (from ptr) and receives (into buf)
            if (ibatch < nproc-1) {
                exchange_sources(ptr, ito, buf, ifrom, xfers);
            }

            // run the O(N^2) calculation concurrently using ptr for sources
            nbody_serial(ptr.n, ptr.x, ptr.y, ptr.ss, ptr.r,
                         numTargs, tx, ty, tax, tay);

            // wait for new data to arrive before continuing
            if (ibatch < nproc-1) {
                // wait on all transfers to complete
                MPI_Waitall(nreqs, xfers, MPI_STATUS_IGNORE);
                // copy buf to work
                work.deep_copy(buf);
                // ptr now points at work
                ptr.shallow_copy(work);
            }
        }

        // and to ensure correct timings, wait for all MPI processes to finish
        MPI_Barrier(MPI_COMM_WORLD);
#else
        // run the O(N^2) calculation
        nbody_serial(numSrcs, src.x, src.y, src.ss, src.r,
                     numTargs, tx, ty, tax, tay);
#endif

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        if (iproc==0) printf("@time of serial run:\t\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        minSerial = std::min(minSerial, elapsed_seconds.count());
    }

    if (test_iterations[3] > 0 and iproc==0) {
        printf("[nbody serial]:\t\t[%.6f] seconds\n", minSerial);
        printf("               \t\t[%.6f] GFlop/s\n", (float)numSrcs*numTargs*num_flops_per*nproc*nproc/(1.e+9*minSerial));

        // Write sample results
        for (int i=0; i<2; i++) printf("   particle %d vel %g %g\n",i,tax[i],tay[i]);
        printf("\n");
    }

#ifdef USE_VC
    if (test_iterations[0] > 0 and test_iterations[3] > 0) {
        // calculate error estimate
        std::vector<float> tax_x86(tax, tax+numTargs);
        float numer = 0.0;
        float denom = 0.0;
        for (size_t i=0; i<tax_vec.size(); ++i) {
            numer += std::pow(tax_vec[i]-tax_x86[i], 2);
            denom += std::pow(tax_x86[i], 2);
        }

        // final echo
        if (iproc==0) {
            printf("\t\t\t(%.3fx speedup using Vc)\n", minSerial/minVc);
            printf("\t\t\t(%.6f RMS error using simd)\n", std::sqrt(numer/denom));
        }
    }
#endif

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
