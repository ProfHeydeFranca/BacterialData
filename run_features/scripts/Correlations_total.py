#From feature_selection.ipynb
#This script calculates Spearman correlations between genomic features
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
if len(sys.argv) < 3:
    print("Usage: python script.py <feature> <abiotic_factor>")
    sys.exit(1)

feature = sys.argv[1]
abiotic_factor = sys.argv[2]

print()

print("Started script! Loading input file...", datetime.datetime.now())
 
#Input
file1 = '/home/bia/Documents/BacterialData/run_features/benchmark_low-variance_threshold/df_' + abiotic_factor + '_' + feature + '_0.002.pickle.zst'  
#file1 = '/work/groups/VEO/shared_data/bia_heyde/df_' + abiotic_factor + '_' + feature + '_selected-filterNA.pickle.zst'  
#Output
file2 = '/home/bia/Documents/BacterialData/run_features/' + abiotic_factor + '/data/spearman_corr_df_' + abiotic_factor + '_' + feature + '_selected-filterNA.pickle.zst'
#file2 = '/work/no58rok/BacterialData/run_features/' + abiotic_factor + '/data/spearman_corr_df_' + abiotic_factor + '_' + feature + '_selected-filterNA.pickle.zst'
file3 = '/home/bia/Documents/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_' + feature + '_selected-filterNA.png' 
#file3 = '/work/no58rok/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_'  + feature + '_selected-filterNA.png' 
file4 = '/home/bia/Documents/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_' + feature + '_selected-filterNA_0.10gap.png'
#file4 = '/work/no58rok/BacterialData/run_features/' + abiotic_factor + '/figures/spearman_corr_df_' + abiotic_factor + '_' + feature + '_selected-filterNA_0.10gap.png'

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
plot_title = 'Correlation of prokaryotes\' ' + feature + ' (' + abiotic_factor + '), n = ' + str(len(X.columns))

#Remove low variance features ########################################

#IMPORTANT: the low-variance features have been previously filtered in step of the pipeline "Remove low-variance features"
#Therefore, for Regression problems, this step is not necessary. 
#However, for classification problems, further isolates are also removed, increasing the number of zero-variance features
print("Remove zero-variance features (should make a difference for Classification only)!")

#Calculate variance for each feature/column of all features
variances = X.var()

#Get list of columns with variance smaller or equal to i
zero_variance_columns = variances[variances <= low_var_threshold].index

#Drop low-variance features
X = X.drop(columns=zero_variance_columns)

print(" Shape of filtered dataframe:", X.shape)
print(" Any NAs in the dataframe?", X.isnull().any().any() )

#Remove low variance features ########################################

#Calculate Spearman correlation########################################

print("Calculating Spearman correlation...", datetime.datetime.now())

#Original line, scipy.stats.spearmanr with pandas df - takes too much memory 
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

if feature != 'kmer9':

    print("Formatting data for plot...", datetime.datetime.now())

    #Populate down_triangle of the matrix and the 1.0 is diagonals with NAs
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

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

if feature != 'kmer9' and feature == 'gene-families':
    
    # Plot the histogram without values close to 0

    print("Plotting values <= -0.1 or >= 0.1 in...", file4, datetime.datetime.now())

    filtered_list = [value for value in cleaned_list if value <= -0.1 or value >= 0.1]

    plt.figure(figsize=(8, 6))
    plt.hist(filtered_list, bins=100, edgecolor='k', alpha=0.7);
    plt.title(plot_title)
    plt.xlabel('Sperman\'s rank correlation value');
    plt.ylabel('Number of correlated pairs');
    plt.savefig(file4, dpi=300)  

print("Finished script!", datetime.datetime.now())
print()
