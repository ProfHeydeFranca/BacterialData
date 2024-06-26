#!/bin/bash

#SBATCH -J join_GF
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --output  join_GF.out.%j
#SBATCH --error   join_GF.err.%j
#SBATCH --mem=240G

source /home/groups/VEO/tools/anaconda3/etc/profile.d/conda.sh
conda activate /home/no58rok/tools/miniconda3/envs/correlation

#Script below used 65GB
#python3 Correlations_total_oxygen.py gene-families

python3 Join_Features_oxygen.py gene-families
