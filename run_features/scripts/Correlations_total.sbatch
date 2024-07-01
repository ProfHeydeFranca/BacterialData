#!/bin/bash

#SBATCH -J corr_GF
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --output  corr_GF.out.%j
#SBATCH --error   corr_GF.err.%j
#SBATCH --mem=10G

conda activate correlation

python3 Correlations_total.py gene-familiesTEST oxygen

