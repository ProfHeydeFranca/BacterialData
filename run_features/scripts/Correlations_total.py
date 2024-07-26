#From feature_selection.ipynb
#This script calculates Spearman correlations between genomic features
#The input is a pickle.zst dataframe with filtered low-variance features
#It outputs a dataframe of feature correlations and 2 histogram plots showing the distribution of correlation values

import sys
import pickle
import zstandard
import pandas as pd
import numpy as np
import datetime
from scipy.stats import rankdata
from scipy.stats import spearmanr

#Get feature from command line
if len(sys.argv) < 4:
    print("Usage: python script.py <feature> <abiotic_factor> <low_var_threshold_and_chunk>")
    sys.exit(1)

feature = sys.argv[1]
abiotic_factor = sys.argv[2]
pre_low_var_threshold = sys.argv[3]
#low_var_threshold can be '0.001' for gene-families, or '0.0002_chunk1' for kmer9 that were split into chunks

#Check if string contains '_chunk'. If so, split the string
if '_chunk0' in pre_low_var_threshold:
    #Split the string
    split_result = pre_low_var_threshold.split('_chunk')
    low_var_threshold = split_result[0]
else:
    low_var_threshold = pre_low_var_threshold

print()

print("Started script! Loading input file...", datetime.datetime.now())
 
#Input
file1 = '/home/bia/Documents/BacterialData/run_features/benchmark_low-variance_threshold/df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + '.pickle.zst'  
#file1 = '/vast/no58rok/BacterialData/run_features/benchmark_low-variance_threshold/df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + '.pickle.zst'  
#Output
file2 = '/home/bia/Documents/BacterialData/run_features/' + abiotic_factor + '/data/spearman_corr_df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + '.pickle.zst'
#file2 = '/vast/no58rok/BacterialData/run_features/' + abiotic_factor + '/data/spearman_corr_df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + 'pickle.zst'
file3 = '/home/bia/Documents/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + '.png' 
#file3 = '/vast/no58rok/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_'  + feature + '_' + pre_low_var_threshold + '.png'
#file4 = '/home/bia/Documents/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + '_0.10gap.png'
#file4 = '/vast/no58rok/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_' + feature + '_' + pre_low_var_threshold + '_0.10gap.png'

with zstandard.open(file1, 'rb') as f:
	df = pickle.load(f)

#Prepare data##########################################################

# Separar as features (X) e o grupos (y)
#Full dataset:
X = df.iloc[:, :-1]  
y = df.iloc[:, -1] 

print(" Shape of initial dataframe:", X.shape)
print(" Any NAs in the dataframe?", X.isnull().any().any() )

#Make plot title (plot is below)
plot_title = 'Subset of 1000 correlation datapoints\' ' + feature + ' (' + abiotic_factor + '), n = ' + str(len(X.columns))

#Remove low variance features ########################################

#IMPORTANT: the low-variance features have been previously filtered in step of the pipeline "Remove low-variance features"
#Therefore, for Regression problems, this step is not necessary. 
#However, for classification problems, further isolates are also removed if a class is removed, increasing the number of zero-variance features
#In these, cases, we will re-apply the low-variance filtering
print("Remove low-variance features (should make a difference for Classification only)!")

#Calculate variance for each feature/column of all features
variances = X.var()

#This if is here to set the variable to 0 in case all chunks have already been joined (should not make a difference)
split_result = low_var_threshold.split('_')

if(split_result[1] == 'chunkAlljoined'):
    low_var_threshold = 0

#Get list of columns with variance smaller or equal to i
zero_variance_columns = variances[variances <= float(low_var_threshold)].index

#Drop low-variance features
X = X.drop(columns=zero_variance_columns)

print(" Shape of filtered dataframe:", X.shape)
print(" Any NAs in the dataframe?", X.isnull().any().any() )

#Remove low variance features ########################################

#Calculate Spearman correlation########################################

print("Calculating Spearman correlation...", datetime.datetime.now())

#scipy.stats.spearmanr with pandas df takes too much memory with a matrix 131,072 x 131,072,
# so we will calculate it for chunks
corr_matrix, _ = spearmanr(X)

#Convert to Pandas DataFrame
column_names = X.columns
correlation_matrix = pd.DataFrame(corr_matrix, columns=column_names, index=column_names)

print(" Shape of the correlation matrix:", correlation_matrix.shape)

print("Saving correlation dataframe...", file2, datetime.datetime.now())

#Save calculations to file
with zstandard.open(file2, 'wb') as f:
	pickle.dump(correlation_matrix, f)
	

#Plot correlation histogram ############################################

print("Subset 1000 datapoints of correlation matrix for plot...", datetime.datetime.now())
#Subset correlation matrix for figure
#If I try to plot all values, I have to use too much memory!
#Subset ~1000 elements

#Get number of columns
n_cols = correlation_matrix.shape[1]

#Determine step size for sampling
step_size = n_cols // 1000

#Subset the DataFrame by taking every step_size-th column
subset_matrix = correlation_matrix.iloc[::step_size, ::step_size]

print("Formatting data for plot...", datetime.datetime.now())

#Populate down_triangle of the matrix and the 1.0 is diagonals with NAs
upper_triangle = subset_matrix.where(np.triu(np.ones(subset_matrix.shape), k=1).astype(bool))

# Flatten the matrix and drop NaN values
correlation_values = upper_triangle.stack().values

# Convert the DataFrame to a NumPy array and flatten it to create a list
values_list = upper_triangle.to_numpy().flatten()

# Remove NaN values from the list
cleaned_list = [value for value in values_list if not np.isnan(value)]

# Check how many values I have
print(" Number of correlation values:",len(cleaned_list))

#Value above should be the same as below
print(" Upper triangle squared minus number of features divided per two:", int( ( (upper_triangle.shape[0] * upper_triangle.shape[1]) - upper_triangle.shape[1] ) / 2 ) )

#Visualize distribution of correlations
import matplotlib.pyplot as plt

print("Plotting all values in...", file3, datetime.datetime.now())

# Plot the histogram full

plt.figure(figsize=(8, 6))
plt.hist(cleaned_list, bins=100, edgecolor='k', alpha=0.7);
plt.title(plot_title)
plt.xlabel('Sperman\'s rank correlation value');
plt.ylabel('Number of correlated pairs');
plt.savefig(file3, dpi=300)  

# Plot the histogram without values close to 0

#print("Plotting values <= -0.1 or >= 0.1 in...", file4, datetime.datetime.now())

#filtered_list = [value for value in cleaned_list if value <= -0.1 or value >= 0.1]

#plt.figure(figsize=(8, 6))
#plt.hist(filtered_list, bins=100, edgecolor='k', alpha=0.7);
#plt.title(plot_title)
#plt.xlabel('Sperman\'s rank correlation value');
#plt.ylabel('Number of correlated pairs');
#plt.savefig(file4, dpi=300)  

print("Finished script!", datetime.datetime.now())
print()
