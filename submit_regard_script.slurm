#!/bin/bash
#SBATCH --job-name=sentiment_cpu
#SBATCH --output=logs/sentiment_%j.out
#SBATCH --error=logs/sentiment_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gender_bias_LLMs

python classify_regard.py