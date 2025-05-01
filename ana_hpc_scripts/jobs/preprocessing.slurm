#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --output=ana_hpc_scripts/logs/outputs/preprocessing_%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/preprocessing_%j.err    

#SBATCH --cpus-per-task=8            # Use 8 CPU cores (adjust if needed)
#SBATCH --mem=64G                    # 32GB RAM (adjust depending on needs)
#SBATCH --time=24:00:00              
#SBATCH --partition=scavenge            
                       
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=aver@itu.dk


# Load Anaconda
module load Anaconda3

# Init conda properly
eval "$(conda shell.bash hook)"


# Activate your environment
conda activate torch_AML

# Change to your project directory if needed
cd /home/aver/AML_MINIPROJECT


# Run the preprocessing script
echo "Starting preprocessing..."
python -m src.data_preprocessing \
    --prod \
    --tokenizer bert-base-uncased \
    --run-name raid_full



