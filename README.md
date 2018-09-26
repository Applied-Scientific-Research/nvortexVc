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

Now, back in this directory you can make all four binaries (2D and 3D, with and without OpenMP).  If the code doesn't build, possibly because your CPU doesn't support the `-mfma` or `-mavx2` options, just edit `Makefile` and remove those.

    make

Run the code with one argument: the number of particles to use.

    nvort2d -n=10000
    nvort2domp -n=10000
    nvort3d -n=10000
    nvort3domp -n=10000

### Performance
I've found that performance of this algorithm accelerated with Vc is very close to that from my [SimdNBodyKernels](https://github.com/markstock/SimdNBodyKernels) code which uses Intel's [ispc](https://github.com/ispc/ispc/) compiler, and the code here is easier to create and understand. One drawback is that Vc will not automatically pad arrays with proper values, so you need to take care to always use multiples of 4 or 8 particles (or 16 whenever Vc supports AVX-512).

On a 16-core Intel i9-7960X Skylake CPU, the nvortex3domp code exceeded 1 TFlop/s - the first time I've ever seen that happen on a single CPU.

