#!/bin/bash

# SLURM script for running experiments on a cluster
#
# This script reads SLURM parameters from a YAML config file and sets up
# the environment to run the experiment. It's designed to be flexible 
# for different cluster configurations.
#
# Usage: sbatch run_slurm.sh config/slurm.yaml
#
# If you need to modify parameters for specific clusters, create a new
# config file or override parameters using environment variables.

# Get the config file from command line
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "No config file provided. Using config/slurm.yaml"
    CONFIG_FILE="config/slurm.yaml"
fi

# Parse partition from config with default
PARTITION=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('partition', 'gpu'))")
JOB_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('job_name', 'factual_recall'))")
NODES=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('nodes', 1))")
NTASKS_PER_NODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('ntasks_per_node', 1))")
CPUS_PER_TASK=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('cpus_per_task', 8))")
GPUS_PER_NODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('gpus_per_node', 1))")
TIME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('time', '48:00:00'))")
MEM=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['slurm'].get('mem', '32G'))")

# Set SLURM parameters
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Create log directory if it doesn't exist
mkdir -p logs

# Get the output directory from config
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['experiment'].get('output_dir', 'outputs/slurm'))")
mkdir -p ${OUTPUT_DIR}

# Check if we need to prepare data first
if [ ! -f "${OUTPUT_DIR}/synthetic_kg.json" ]; then
    echo "Preparing data first..."
    python scripts/prepare_data.py --config "$CONFIG_FILE"
fi

# Print environment information
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "Python executable: $(which python)"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment: $VIRTUAL_ENV"
fi

# Use the appropriate CUDA device if available
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
fi

# Load modules if needed
# module load python/3.10 cuda/11.8

# Activate virtual environment if needed
# Use # conda activate your_env  # for conda
# Or  # source /path/to/venv/bin/activate  # for virtualenv

# Set environment variables
export PYTHONUNBUFFERED=1  # Ensure Python output is unbuffered
export TOKENIZERS_PARALLELISM=false  # Prevent tokenizers warning

# Run experiment with timing information
echo "Starting experiment at $(date)"
start_time=$(date +%s)

python scripts/run_experiment.py --config "$CONFIG_FILE"
exit_code=$?

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

echo "Experiment finished at $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Exit code: $exit_code"

exit $exit_code
