#!/bin/bash

#SBATCH -J teste
#SBATCH --partition=gpu
#SBATCH --ntasks=30
#SBATCH --output  test.out.%j
#SBATCH --error   test.err.%j
#SBATCH --time=2-00:00:00
#SBATCH --mem=400G

source /home/groups/VEO/tools/anaconda3/etc/profile.d/conda.sh
conda activate /home/no58rok/tools/miniconda3/envs/bacterial_phenotypes/

python3 teste.py
