#!/bin/bash
#
#SBATCH --job-name=cosipy_aws
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00 
#SBATCH --mail-user=simon.goetz@student.uibk.ac.at
#SBATCH --qos=normal

# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e



# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
COSIPY_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/wd"
mkdir -p "$COSIPY_WORKDIR"
export COSIPY_WORKDIR
echo "Workdir for this run: $COSIPY_WORKDIR"


# Add other useful defaults
export LRU_MAXSIZE=1000

COSIPY_OUTDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/out"
mkdir -p "$COSIPY_OUTDIR"
export COSIPY_OUTDIR
echo "Output dir for this run: $COSIPY_OUTDIR"


# All commands in the EOF block run inside of the container
# Adjust container version to your needs, they are guaranteed to never change after their respective day has passed.
srun python COSIPY.py

# Write out
echo "Copying files..."
rsync -avzh "$COSIPY_OUTDIR/" test_output
# rsync -avz --no-perms --no-owner --no-group "$COSIPY_OUTDIR/" output

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"