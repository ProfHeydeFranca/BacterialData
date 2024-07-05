#!/bin/bash

#SBATCH -J corr_XX
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --output  corr_XX.out.%j
#SBATCH --error   corr_XX.err.%j
#SBATCH --mem=60G

module purge
source /home/groups/VEO/tools/anaconda3/etc/profile.d/conda.sh
conda activate /home/no58rok/tools/miniconda3/envs/correlation

python3 Correlations_total.py XX XX

conda deactivate
