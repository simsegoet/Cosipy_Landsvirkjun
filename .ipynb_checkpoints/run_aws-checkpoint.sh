#!/bin/bash
#
#SBATCH --job-name=cosipy_aws
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00 
#SBATCH --mail-user=simon.goetz@student.uibk.ac.at
#SBATCH --qos=normal

# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e



# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
AWS_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/wd"
mkdir -p "$AWS_WORKDIR"
export AWS_WORKDIR
echo "Workdir for this run: $AWS_WORKDIR"


# Add other useful defaults
export LRU_MAXSIZE=1000




# All commands in the EOF block run inside of the container
# Adjust container version to your needs, they are guaranteed to never change after their respective day has passed.
srun python -m cosipy.utilities.aws2cosipy.aws2cosipy     -i ./data/input/Bruarjokull/B13_final_all.csv     -o ./data/input/Bruarjokull/B16_full_glacier222_2001_2020.nc     -s ./data/static/Bruarjokull_static_test.nc     -b 20010101 -e 20201231



# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"