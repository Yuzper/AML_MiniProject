#!/bin/bash

#SBATCH --job-name=preprocessing-ana
#SBATCH --output=ana_hpc_scripts/logs/outputs/preprocessing_%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/preprocessing_%j.err    # Optional: separate stderr log
#SBATCH --gres=gpu  
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-12:00:00
#SBATCH --partition=scavenge
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=aver@itu.dk 

# Activate the virtual environment
source .venv/bin/activate

echo "Running on $(hostname):"

# Set PYTHONPATH to include the repository root
export PYTHONPATH=$(pwd)

# check enviroment 
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"

# Run the preprocessing script
echo "Starting preprocessing..."
python -u -m src.data_preprocessing --run-name raid_full --prod



