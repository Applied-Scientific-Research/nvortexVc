CC=g++
#CPPOPTS=-O3 -mavx2 -mfma
CPPOPTS=-O3 -march=native
INCLUDE=-I/opt/Vc/include
LIBS=-L/opt/Vc/lib -lVc

all : nvortex2d nvortex2domp nvortex3d nvortex3domp

nvortex2d : nvortex2d.cpp
	$(CC) $(CPPOPTS) $(INCLUDE) -o $@ $< $(LIBS)
nvortex2domp : nvortex2d.cpp
	$(CC) $(CPPOPTS) $(INCLUDE) -fopenmp -o $@ $< $(LIBS)
nvortex3d : nvortex3d.cpp
	$(CC) $(CPPOPTS) $(INCLUDE) -o $@ $< $(LIBS)
nvortex3domp : nvortex3d.cpp
	$(CC) $(CPPOPTS) $(INCLUDE) -fopenmp -o $@ $< $(LIBS)

clean :
	rm -f nvortex2d nvortex2domp nvortex3d nvortex3domp
