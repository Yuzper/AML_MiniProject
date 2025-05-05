#!/bin/bash

#SBATCH --job-name=evaluation
#SBATCH --output=ana_hpc_scripts/logs/outputs/evaluation_%j.out   # Log file (%j = job ID)
#SBATCH --error=ana_hpc_scripts/logs/errors/evaluation_%j.err    

#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                    

#SBATCH --time=10:00:00              
#SBATCH --partition=scavenge                                    
                 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=aver@itu.dk


CONTAINER=/opt/itu/containers/pytorchtransformers/pytorch-24.07-py3-transformers.sif
BIND_PATH=/home/aver/Machine_Learning/AML_MiniProject

echo "Checking Python version in container:"
singularity exec $CONTAINER python --version

echo "Starting job inside Singularity container"

# Quick check that your installed packages are visible
singularity exec -B $BIND_PATH:/mini_project $CONTAINER \
    python -c "import transformers; import datasets; import evaluate; print(transformers.__version__)"

# Run your script
singularity exec -B $BIND_PATH:/mini_project $CONTAINER \
    bash -c "cd /mini_project && python3 -m src.evaluate \
        --dataset-root data/processed/raid_full \
        --run-name bert_full \
        --checkpoints-dir outputs/checkpoints/bert_full/checkpoint-74718 \
        --split val,test"


echo "Job completed"
