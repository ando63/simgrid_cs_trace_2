#define CLASS 'W'
#define NUM_PROCS 16
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   */
   
#define NUM_SAMPLES 13824
#define STD_DEVIATION 256
#define NUM_SOURCES 8
#define COMPILETIME "23 Jul 2024"
#define NPBVERSION "3.3.1"
#define MPICC "${SIMGRID_PATH}/bin/smpicc"
#define CFLAGS "-O"
#define CLINK "$(MPICC)"
#define CLINKFLAGS "-O -L${SIMGRID_PATH}/lib"
#define CMPI_LIB "-lsimgrid -lgfortran #-lsmpi -lgras"
#define CMPI_INC "-I${SIMGRID_PATH}/include/smpi/"
