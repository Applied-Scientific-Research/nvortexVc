# nvortexVc
Direct Biot-Savart solver for 2D and 3D vortex blobs accelerated with Vc

### Build and run
This should be easy on a Linux host, but first you will need to build and install [Vc](https://github.com/VcDevel/Vc).

    git clone https://github.com/VcDevel/Vc.git
    cd Vc
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/opt/Vc -DBUILD_TESTING=OFF ..
    make -j 4
    sudo make install

Now, back in this directory you can make all four binaries (2D and 3D, with and without OpenMP).

    make

Run the code with one argument: the number of particles to use.

    nvortex2d -n=10000
    nvortex2domp -n=10000
    nvortex3d -n=10000
    nvortex3domp -n=10000

### Performance
Performance of this algorithm accelerated with Vc is slightly better than from my [SimdNBodyKernels](https://github.com/markstock/SimdNBodyKernels) code which uses Intel's [ispc](https://github.com/ispc/ispc/) compiler, and the code here is easier to create and understand. A little care must be taken to pad arrays with proper values (padded particle radius must be nonzero), lest the inner kernel try to divide by zero when using the last vector register set of particles.

On a 16-core Intel i9-7960X Skylake CPU, with a large-enough problem, the `nvortex3domp` code exceeded 1 TFlop/s - the first time I've ever seen that happen on a single CPU. The 16-core Ryzen 9 achieved 1133 GFlop/s with `./nvortex3domp -n=100000`.

The table below gives peak performance in GFlop/s for the 3D case (with no grads) for combinations of x86 (possibly auto-vectorized) vs. Vc, and single-thread vs. OpenMP multithreaded. Each was run on a Fedora or Ubuntu machine and compiled with GCC (7.2, 7.3, 9.3.1, 11.2). The numbers are all GFlop/s and the problem size was 30000 (`nvortex3d -n=30000`).

| GFlop/s           | x86 serial | x86 OMP | Vc serial | Vc OMP | CPU peak | % of theoretical |
|-------------------|------------|---------|-----------|--------|----------|------------------|
| Intel i3-8109U    |     5.2    |   11.5  |   69.6    |  145   |   230.4  |        63%       |
| Intel i7-7500U    |     5.1    |   11.0  |   62.2    |  121   |   204.8  |        59%       |
| Intel i7-5960X    |     4.3    |   34.0  |   44.0    |  369   |   768    |        48%       |
| Intel i9-7960X    |     4.3    |   64.5  |   58.5    |  813   |  1434    |        57%       |
| AMD Ryzen 7 2700X |     4.5    |   47.3  |   42.9    |  289   |   499.2  |        58%       |
| AMD Ryzen 9 3950X |     6.2    |   62.3  |   86.6    |  790   |  2048    |        39%       |

Note that the peak flops rate depends on problem size, number of cores, and cache sizes. For example, the AMD Ryzen 9 3950X can achieve 1154 GFlop/s (56% of theoretical peak) with `nvortex3domp -n=150000`.

