#!/bin/bash

#SBATCH --job-name=train_base_mode-ana
#SBATCH --output=ana_hpc_scripts/logs/outputs/train_base_model_%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/train_base_mode_%j.err    # Optional: separate stderr log

#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                    
#SBATCH --nodes=1
#SBATCH --nodelist=cn[3,4,5,6,7,9,10,12,13,18,19]
#SBATCH --gres=gpu

#SBATCH --time=24:00:00              
#SBATCH --partition=acltr                                    
        
#SBATCH --time=0-30:00:00            
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=aver@itu.dk



module load Anaconda3

# Init conda properly
eval "$(conda shell.bash hook)"

conda activate tensor


python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


python -m src.train \
       --dataset-root data/processed/raid_full \
       --model-name  bert-base-uncased \
       --epochs 5 \
       --run-name   raid_full_gpu