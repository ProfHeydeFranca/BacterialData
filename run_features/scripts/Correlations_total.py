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
#import xarray as xr
#import bottleneck
#import dask.dataframe as dd

#Get feature from command line
if len(sys.argv) < 3:
    print("Usage: python script.py <feature> <abiotic_factor>")
    sys.exit(1)

feature = sys.argv[1]
abiotic_factor = sys.argv[2]

print()

print("Started script! Loading input file...", datetime.datetime.now())
 
#Input
file1 = '/home/bia/Documents/bacterial_phenotypes/connecting_features_abFactors/df_' + abiotic_factor + '_' + feature + '_selected-filterNA.pickle.zst'  
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

print(" Shape of the input dataframe:", df.shape)

# Separar as features (X) e o grupos (y)
#Full dataset:
X = df.iloc[:, :-1]  
y = df.iloc[:, -1] 

#Remove zero-variance features#########################################
#NEW BLOCK TO REMOVE ZERO-VARIANCE FEATURES

print("Removing zero-variance features...", datetime.datetime.now())

#Calculate variance for each feature/column
variances = X.var()

#Identify columns with zero variance
zero_variance_columns = variances[variances == 0].index
#print(zero_variance_columns)

print("Number of zero-variance features:", len(zero_variance_columns), ', or:', round( (len(zero_variance_columns)/len(X.columns))*100, 1 ), '% of the total of features')

#Drop zero-variance features
X = X.drop(columns=zero_variance_columns)

print(" Shape of dataframe without zero-variance features:", X.shape)
print(" Any NAs in the dataframe?", X.isnull().any().any() )



#Make plot title (plot is below)
plot_title = 'Correlation of prokaryotes\' ' + feature + ' (' + abiotic_factor + '), n = ' + str(len(X.columns))

#Calculate Spearman correlation########################################

print("Calculating Spearman correlation...", datetime.datetime.now())

#Approach 1 - slow
############################ pandas calculation, very slow
# Calcular a correlação de Spearman
#correlation_matrix = X.corr(method='spearman')

#Approach 2 - segFault with large input
############################ ChatGPT: Spearman correlation using numpy for more efficiency, segmentation fault with large kmer input

#Function to calculate Spearman's rank correlation using numpy
#def spearman_correlation(data):
    
    # Save column names
#    column_names = data.columns

#    print("Part 1")
    # Rank the data
#    ranked_data = np.apply_along_axis(rankdata, 0, data)

#    print("Part 2")
    # Calculate the Pearson correlation on the ranked data
#    ranked_data -= ranked_data.mean(axis=0)  # Center the data by subtracting the mean

#    print("Part 3")
    #print("Part 3", ranked_data[0:3])
#    print("Part 3.1")
#    tmp = np.dot(ranked_data.T, ranked_data)

#    print("Part 3.2")
#    cov_matrix = tmp / (ranked_data.shape[0] - 1)
 
####    cov_matrix = np.dot(ranked_data.T, ranked_data) / (ranked_data.shape[0] - 1)

#    print("Part 4", cov_matrix[0:3], ranked_data.shape[0])    
#    std_devs = np.sqrt(np.diag(cov_matrix))

#    print("Part 5")
#    spearman_corr = cov_matrix / np.outer(std_devs, std_devs)

#    print("Part 6")
    # Convert the correlation matrix to a DataFrame
#    spearman_corr_df = pd.DataFrame(spearman_corr, columns=column_names, index=column_names)

#    print("Part 7")
#    return spearman_corr_df

#correlation_matrix = spearman_correlation(X)

#Approach 3 - on use
############################ ChatGPT: Spearman correlation using scipy.stats (which uses numpy internally)

#import pandas as pd
#from scipy.stats import spearmanr

#Prepare data using DASK###############################################

#Calculate Spearman correlation of numpy array

#Original line, scipy.stats.spearmanr with pandas df - takes too much memory 
#corr_matrix, _ = spearmanr(X)

#Same as above, but numpy array as input to function - improvement in memory and speed is very small
X_array = X.values
corr_matrix, _ = spearmanr(X_array) 

# Convert to Pandas DataFrame
column_names = X.columns
correlation_matrix = pd.DataFrame(corr_matrix, columns=column_names, index=column_names)

#Convert to Dask DataFrame ###########################

#Generate correlation matrix using Pearson - takes too long ############ START
#corr_matrix = np.zeros((X.shape[1],X.shape[1]))
#corr_matrix = dX.corr(method='pearson')
#corr_matrix = corr_matrix.compute()
#Generate correlation matrix using Pearson -takes too long ############ END

# Convert X to Dask DataFrame
#dX = dd.from_pandas(X, npartitions=10)

# Step 2: Compute Spearman correlation using Dask

# Step 1: Convert Pandas DataFrame to xarray.DataArray
#data_array = xr.DataArray(X)

#chunked1 = array1.chunk({'place': 10})
#chunked2 = array2.chunk({'place': 10})

#def covariance_gufunc(x, y):
#    return ((x - x.mean(axis=-1, keepdims=True))
#            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

#def pearson_correlation_gufunc(x, y):
#    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

#def spearman_correlation_gufunc(x, y):
#    x_ranks = bottleneck.rankdata(x, axis=-1)
#    y_ranks = bottleneck.rankdata(y, axis=-1)
#    return pearson_correlation_gufunc(x_ranks, y_ranks)

#def spearman_correlation(x, y, dim):
#    return xr.apply_ufunc(
#        spearman_correlation_gufunc, x, y,
#        input_core_dims=[[dim], [dim]],
#        dask='parallelized',
#        output_dtypes=[float])

#corr_matrix = spearman_correlation(chunked1, chunked2).compute()
# Step 4: Compute Spearman correlation matrix
#corr_matrix = spearman_correlation(data_array, data_array, ).compute()


# Convert to Pandas DataFrame
#column_names = X.columns
#correlation_matrix = pd.DataFrame(corr_matrix, columns=column_names, index=column_names)

############################


print(" Shape of the correlation matrix:", correlation_matrix.shape)

print("Saving correlation dataframe...", file2, datetime.datetime.now())

#Save calculations to file
with zstandard.open(file2, 'wb') as f:
	pickle.dump(correlation_matrix, f)
	
	

#Plot correlation histogram############################################

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
