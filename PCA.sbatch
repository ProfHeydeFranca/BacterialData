#!/bin/bash

#SBATCH -J PCA
#SBATCH --partition=gpu
#SBATCH --ntasks=30
#SBATCH --output  pca.out.%j
#SBATCH --error   pca.err.%j
#SBATCH --time=2-00:00:00
#SBATCH --mem=400G

source /home/groups/VEO/tools/anaconda3/etc/profile.d/conda.sh
conda acvtivate /home/wi38kap/.conda/envs/bacterial_phenotypes

python3 PCA_Without_class.py
