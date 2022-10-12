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
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

static float num_flops_per = 30.f;

// serial (x86) instructions

static inline void nbody_kernel_serial(const float sx, const float sy, const float sz,
                                       const float ssx,const float ssy,const float ssz, const float sr,
                                       const float tx, const float ty, const float tz, const float tr,
                                       float* const __restrict__ tax, float* const __restrict__ tay, float* const __restrict__ taz) {
    // 30 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    float r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr*tr;
    r2 = 1.0/(r2*sqrt(r2));
    *tax += r2 * (dz*ssy - dy*ssz);
    *tay += r2 * (dx*ssz - dz*ssx);
    *taz += r2 * (dy*ssx - dx*ssy);
}

void nbody_serial(const int numSrcs, const float* const __restrict__ sx,
                                     const float* const __restrict__ sy,
                                     const float* const __restrict__ sz,
                                     const float* const __restrict__ ssx,
                                     const float* const __restrict__ ssy,
                                     const float* const __restrict__ ssz,
                                     const float* const __restrict__ sr,
                  const int numTarg, const float* const __restrict__ tx,
                                     const float* const __restrict__ ty,
                                     const float* const __restrict__ tz,
                                     const float* const __restrict__ tr,
                                     float* const __restrict__ tax,
                                     float* const __restrict__ tay,
                                     float* const __restrict__ taz) {

    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        for (int j = 0; j < numSrcs; j++) {
            nbody_kernel_serial(sx[j], sy[j], sz[j], ssx[j], ssy[j], ssz[j], sr[j],
                                tx[i], ty[i], tz[i], tr[i], &tax[i], &tay[i], &taz[i]);
        }
    }
}

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
  float* z = nullptr;
  float* r = nullptr;
  float* sx = nullptr;
  float* sy = nullptr;
  float* sz = nullptr;

  // sized constructor
  Sources(const int _n) : n(_n) {
    x = new float[_n];
    y = new float[_n];
    z = new float[_n];
    r = new float[_n];
    sx = new float[_n];
    sy = new float[_n];
    sz = new float[_n];
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
    for (int i=0; i<n; ++i) z[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) sx[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) sy[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) sz[i] = zmean_dist(_gen);
    const float rad = 1.0 / std::sqrt((float)n);
    for (int i=0; i<n; ++i) r[i] = rad;
  }

  void deep_copy(const Sources& _s) {
    assert(n == _s.n && "Copying from Sources with different n");
    for (int i=0; i<n; ++i) x[i] = _s.x[i];
    for (int i=0; i<n; ++i) y[i] = _s.y[i];
    for (int i=0; i<n; ++i) z[i] = _s.z[i];
    for (int i=0; i<n; ++i) r[i] = _s.r[i];
    for (int i=0; i<n; ++i) sx[i] = _s.sx[i];
    for (int i=0; i<n; ++i) sy[i] = _s.sy[i];
    for (int i=0; i<n; ++i) sz[i] = _s.sz[i];
  }

  void shallow_copy(const Sources& _s) {
    // warning: this can orphan memory!!!
    n = _s.n;
    x = _s.x;
    y = _s.y;
    z = _s.z;
    r = _s.r;
    sx = _s.sx;
    sy = _s.sy;
    sz = _s.sz;
  }

  // destructor
  ~Sources() { }
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
  MPI_Irecv(recvsrcs.z, recvsrcs.n, MPI_FLOAT, ifrom, 12, MPI_COMM_WORLD, &handle[4]);
  MPI_Isend(sendsrcs.z, sendsrcs.n, MPI_FLOAT, ito, 12, MPI_COMM_WORLD, &handle[5]);
  MPI_Irecv(recvsrcs.r, recvsrcs.n, MPI_FLOAT, ifrom, 13, MPI_COMM_WORLD, &handle[6]);
  MPI_Isend(sendsrcs.r, sendsrcs.n, MPI_FLOAT, ito, 13, MPI_COMM_WORLD, &handle[7]);
  MPI_Irecv(recvsrcs.sx, recvsrcs.n, MPI_FLOAT, ifrom, 14, MPI_COMM_WORLD, &handle[8]);
  MPI_Isend(sendsrcs.sx, sendsrcs.n, MPI_FLOAT, ito, 14, MPI_COMM_WORLD, &handle[9]);
  MPI_Irecv(recvsrcs.sy, recvsrcs.n, MPI_FLOAT, ifrom, 15, MPI_COMM_WORLD, &handle[10]);
  MPI_Isend(sendsrcs.sy, sendsrcs.n, MPI_FLOAT, ito, 15, MPI_COMM_WORLD, &handle[11]);
  MPI_Irecv(recvsrcs.sz, recvsrcs.n, MPI_FLOAT, ifrom, 16, MPI_COMM_WORLD, &handle[12]);
  MPI_Isend(sendsrcs.sz, sendsrcs.n, MPI_FLOAT, ito, 16, MPI_COMM_WORLD, &handle[13]);
}
#endif

static void usage() {
    fprintf(stderr, "Usage: nvortex3d.bin [-n=<number>] [simd iterations] [serial iterations]\n");
    exit(1);
}

// main program

int main(int argc, char *argv[]) {

    static unsigned int test_iterations[] = {1, 1, 1, 1};
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
    const int nreqs = 14;
    MPI_Request* xfers = new MPI_Request[nreqs];
    for (int ix=0; ix<nreqs; ++ix) xfers[ix] = MPI_REQUEST_NULL;
    //std::cout << "  proc " << iproc << " always sending to " << ito << " and receiving from " << ifrom << std::endl;
#endif

    // allocate target particle data
    //Targets trg = Targets(numTargs);
    //trg.init_rand(gen);
    float* tx = new float[numTargs];
    float* ty = new float[numTargs];
    float* tz = new float[numTargs];
    float* tr = new float[numTargs];
    float* tax = new float[numTargs];
    float* tay = new float[numTargs];
    float* taz = new float[numTargs];
    for (int i=0; i<numTargs; i++) tx[i] = zmean_dist(gen);
    for (int i=0; i<numTargs; i++) ty[i] = zmean_dist(gen);
    for (int i=0; i<numTargs; i++) tz[i] = zmean_dist(gen);
    for (int i=0; i<numTargs; i++) tr[i] = 1.0 / std::sqrt((float)numTargs);

    //
    // And run the serial implementation a few times, again reporting the minimum
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[3]; ++i) {
        auto start = std::chrono::system_clock::now();

        // zero the vels
        for (int j=0; j<numTargs; ++j) tax[j] = 0.0;
        for (int j=0; j<numTargs; ++j) tay[j] = 0.0;
        for (int j=0; j<numTargs; ++j) taz[j] = 0.0;

#ifdef USE_MPI
        // first set to compute is self
        ptr.shallow_copy(src);

        for (int ibatch=0; ibatch<nproc ; ++ibatch) {
            // post non-blocking sends (from ptr) and receives (into buf)
            if (ibatch < nproc-1) {
                exchange_sources(ptr, ito, buf, ifrom, xfers);
            }

            // run the O(N^2) calculation concurrently using ptr for sources
            nbody_serial(ptr.n, ptr.x, ptr.y, ptr.z, ptr.sx, ptr.sy, ptr.sz, ptr.r,
                         numTargs, tx, ty, tz, tr, tax, tay, taz);

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
        nbody_serial(numSrcs, src.x, src.y, src.z, src.sx, src.sy, src.sz, src.r,
                     numTargs, tx, ty, tz, tr, tax, tay, taz);
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
        for (int i=0; i<2; i++) printf("   particle %d vel %g %g %g\n",i,tax[i],tay[i],taz[i]);
        printf("\n");
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
