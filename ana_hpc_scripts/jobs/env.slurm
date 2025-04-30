#!/bin/bash

#SBATCH --job-name=environment
#SBATCH --output=hpc_slurms/logs/outputs/env_%j.out   # Log file (%j = job ID)
#SBATCH --error=hpc_slurms/logs/errors/envs_%j.err    

#SBATCH --cpus-per-task=8            # Use 8 CPU cores (adjust if needed)
#SBATCH --mem=32G                    # 32GB RAM (adjust depending on needs)
#SBATCH --time=3:00:00              
#SBATCH --partition=cores            # Because your project might use GPU
                       
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=aver@itu.dk


echo "Running on $(hostname):"
module load Anaconda3
conda create --name torch_AML
source activate torch_AML
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas scikit-learn matplotlib seaborn jupyterlab ipykernel
pip install click datasets evaluate transformers accelerate langchain nltk tqdm tokenizers dataclasses-json orjson rouge-score loguru

conda env export > environment.yaml