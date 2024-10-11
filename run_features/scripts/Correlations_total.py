#From feature_selection-development-only.ipynb
#This script calculates Spearman correlations between genomic features
#The input is the output of filtering based on variance
#The input is a pickle.zst dataframe with correlation values

import sys
import pickle
import zstandard
import pandas as pd
import numpy as np
import datetime
from scipy.stats import rankdata
from scipy.stats import spearmanr

#Usage
#python script.py

print()
print("Started script! Loading input file...", datetime.datetime.now())
 
#Input
path = '/home/bia/Documents/BacterialData/run_features/benchmark_low-variance_threshold/'
file = path + 'df_salt_gene-familiesClassification_0.009000000000000001final-filtering.pickle.zst' 

#Output
path_out = '/home/bia/Documents/BacterialData/run_features/join_highly_correlated/'
file_out = path_out + 'df_corr_salt_gene-familiesClassification.pickle.zst'

with zstandard.open(file, 'rb') as f:
	df = pickle.load(f)

#Prepare data##########################################################

# Separar as features (X) e o grupos (y)
#Full dataset:
X = df.iloc[:, :-1]  
y = df.iloc[:, -1] 

print(" Shape of initial dataframe:", X.shape)
print(" Any NAs in the dataframe?", X.isnull().any().any() )

#Calculate Spearman correlation########################################

print("Calculating Spearman correlation...", datetime.datetime.now())

#scipy.stats.spearmanr only works with relatively small input files
corr_matrix, _ = spearmanr(X)

#Convert to Pandas DataFrame
column_names = X.columns
correlation_matrix = pd.DataFrame(corr_matrix, columns=column_names, index=column_names)

print(" Shape of the correlation matrix:", correlation_matrix.shape)

print("Saving correlation dataframe...", file_out, datetime.datetime.now())

#Save calculations to file
with zstandard.open(file_out, 'wb') as f:
	pickle.dump(correlation_matrix, f)
	
print("Finished script!", datetime.datetime.now())
print()