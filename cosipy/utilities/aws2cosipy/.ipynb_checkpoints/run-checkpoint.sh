#!/bin/bash -l

# The batch system should use the current directory as working directory.
#SBATCH --job-name=REAL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=03:00:00

module load intel64 netcdf 

export KMP_STACKSIZE=64000000
export OMP_NUM_THREADS=1
ulimit -s unlimited

 
python3 aws2cosipy.py     -i ../../data/input/Bruarjokull/B13_all_rdy_conv_elaboretemweq.csv     -o ../../data/input/Bruarjokull/Bruarjokull_2008_2018_60_mult.nc     -s ../../data/static/Bruarjokull_static_final2.nc     -b 20080101 -e 20181231