#!/bin/bash

#SBATCH --job-name=train_base_mode-ana
#SBATCH --output=ana_hpc_scripts/logs/outputs/train_base_model_%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/train_base_mode_%j.err    # Optional: separate stderr log

#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                    

#SBATCH --time=2:00:00              
#SBATCH --partition=scavenge                                    
                 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=aver@itu.dk


# Load Anaconda
module load Anaconda3

# Init conda properly
eval "$(conda shell.bash hook)"


# Activate your environment
conda activate torch_1


python -m src.train_torch \
       --dataset-root data/processed/raid_full \
       --model-name bert-base-uncased \
       --epochs 5 \
       --batch-size 8 \
       --learning-rate 2e-5 \
       --run-name bert_full