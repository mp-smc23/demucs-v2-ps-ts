#!/bin/bash
#SBATCH --job-name=res     # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=24G                       # Memory
#SBATCH --time=12:00:00 # Timeout
#SBATCH --signal=B:SIGTERM@30


#####################################################################################

# tweak this to fit your needs
max_restarts=7

# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, it's the first run, so set restarts to 0
iteration=${restarts:-0}

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/res_${SLURM_JOB_ID}/log_${iteration}.out"
errfile="logs/res_${SLURM_JOB_ID}/log_${iteration}.err"

mkdir -p logs/res_${SLURM_JOB_ID}

##  Define a term-handler function to be executed           ##
##  when the job gets the SIGTERM (before timeout)          ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
        # Requeue the job, allowing it to restart with incremented iteration
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler when the job gets terminated
trap 'term_handler' SIGTERM

#######################################################################################


# Run Singularity and execute commands inside the container

srun --output="${outfile}" --error="${errfile}" singularity exec --nv ./demucs.sif python3 run.py --test pitsisdr-res.th -b 128 -e 150 --PITSISDR --repeat 1 --repitch 0 --audio_channels 1 --wav /ceph/home/student.aau.dk/xg64zo/smc10/res --musdb /ceph/home/student.aau.dk/xg64zo/smc10/noaug
# srun --output="${outfile}" --error="${errfile}" singularity exec --nv ./demucs.sif python3 -m demucs -b 4 -e 1 --PITSISDR --repeat 1 --wav /ceph/home/student.aau.dk/xg64zo/smc10/res --musdb /ceph/home/student.aau.dk/xg64zo/smc10/res
