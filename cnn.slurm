#!/bin/bash
#SBATCH --job-name=global_cnn          # Job name
#SBATCH --output=output.txt            # Output file (%j will be replaced with the job ID)
#SBATCH --error=error.txt              # Error file
#SBATCH --nodes=10                     # Number of nodes
#SBATCH --ntasks-per-node=8            # Number of tasks per node (MPI processes per node)
#SBATCH --time=01:00:00                # Time limit (hh:mm:ss)
#SBATCH -p short

# Load the necessary modules
module load anaconda/3                         # Replace with the correct Anaconda module on your system
module load openmpi/gcc13.1.0/4.1.5            # Load OpenMPI module

# Activate your Anaconda environment
source activate /lustre/home/user/.conda/envs/mpi_env

# Run the MPI program
mpirun -np $SLURM_NTASKS python final_mpi.py