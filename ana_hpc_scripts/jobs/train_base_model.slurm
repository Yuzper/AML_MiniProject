#!/bin/bash

#SBATCH --job-name=train_base_model
#SBATCH --output=ana_hpc_scripts/logs/outputs/train_base_model_%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/train_base_model_%j.err    

#SBATCH --nodes=1  
#SBATCH --nodelist=cn[7,9,11,13,18]      
#SBATCH --gres=gpu 
#SBATCH --mem=30G                    

#SBATCH --time=5:00:00              
#SBATCH --partition=acltr                                    
                 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=aver@itu.dk

CONTAINER=/opt/itu/containers/pytorchtransformers/pytorch-24.07-py3-transformers.sif
BIND_PATH=/home/aver/Machine_Learning/AML_MiniProject
MOUNT_POINT=/mini_project   

echo "Python inside container:"
singularity exec --nv "$CONTAINER" python --version

echo "GPU(s) visible in container:"
singularity exec --nv "$CONTAINER" nvidia-smi


echo "Checking if torch can see GPU (inside container):"
singularity exec --nv $CONTAINER python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Device:", torch.cuda.get_device_name(0))'

echo "Starting job inside Singularity container"

# ---------------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------------
singularity exec --nv -B "$BIND_PATH":"$MOUNT_POINT" "$CONTAINER" \
    bash -c "
        cd $MOUNT_POINT &&
        python -m src.train_torch \
            --dataset-root data/processed/raid_full \
            --model-name bert-base-uncased \
            --epochs 3 \
            --batch-size 8 \
            --learning-rate 2e-5 \
            --run-name bert_full
    "

echo "Training job finished"