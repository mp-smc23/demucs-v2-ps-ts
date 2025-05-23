#!/bin/bash
#SBATCH --job-name=test_model                # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:1                    # Request 8 GPU
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=24G                       # Memory
#SBATCH --time=12:00:00 # Timeout

#####################################################################################

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/noaug_${SLURM_JOB_ID}/log.out"
errfile="logs/noaug_${SLURM_JOB_ID}/log.err"

mkdir -p logs/noaug_${SLURM_JOB_ID}

#######################################################################################


# Run Singularity and execute commands inside the container

srun --output="${outfile}" --error="${errfile}" singularity exec --nv ./demucs.sif python3 -m demucs --test mads-noaug.th -b 16 --repeat 1 --repitch 0 --audio_channels 1 --wav /ceph/home/student.aau.dk/xg64zo/smc10/mad --musdb /ceph/home/student.aau.dk/xg64zo/smc10/old_convs/noaug