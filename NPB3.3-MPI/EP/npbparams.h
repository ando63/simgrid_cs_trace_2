c NPROCS = 16 CLASS = W
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  
        character class
        parameter (class ='W')
        integer m, npm
        parameter (m=25, npm=16)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='23 Jul 2024')
        character*5 npbversion
        parameter (npbversion='3.3.1')
        character*26 cs1
        parameter (cs1='${SIMGRID_PATH}/bin/smpiff')
        character*9 cs2
        parameter (cs2='$(MPIF77)')
        character*35 cs3
        parameter (cs3='-lsimgrid -lgfortran #-lsmpi -lgras')
        character*31 cs4
        parameter (cs4='-I${SIMGRID_PATH}/include/smpi/')
        character*2 cs5
        parameter (cs5='-O')
        character*24 cs6
        parameter (cs6='-O -L${SIMGRID_PATH}/lib')
        character*6 cs7
        parameter (cs7='randi8')
