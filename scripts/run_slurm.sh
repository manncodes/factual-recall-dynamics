#!/bin/bash

# SLURM script for running experiments on a cluster
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Get the config file from command line
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "No config file provided. Using default.yaml"
    CONFIG_FILE="config/slurm.yaml"
fi

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Run the experiment
python scripts/run_experiment.py --config "$CONFIG_FILE"
