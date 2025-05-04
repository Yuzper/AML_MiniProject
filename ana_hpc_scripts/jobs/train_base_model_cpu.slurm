#!/bin/bash

#SBATCH --job-name=train_roberta
#SBATCH --output=ana_hpc_scripts/logs/outputs/train_roberta%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/train_roberta%j.err    # Optional: separate stderr log

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
conda activate torch_AML


python -m src.train_torch \
       --dataset-root data/processed/raid_full \
       --model-name roberta-base \
       --epochs 5 \
       --batch-size 8 \
       --learning-rate 2e-5 \
       --run-name roberta-base
       