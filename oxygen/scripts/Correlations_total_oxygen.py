#From feature_selection.ipynb

import sys
import pickle
import zstandard
import numpy as np
import datetime

#Get feature from command line
if len(sys.argv) < 2:
    print("Usage: python script.py <feature>")
    sys.exit(1)

feature = sys.argv[1]


print()

print("Started script! Loading input file...", datetime.datetime.now())
 
#Input
#file1 = '/home/bia/Documents/bacterial_phenotypes/connecting_features_abFactors/df_oxygen_' + feature + '_selected-filterNA.pickle.zst'  
file1 = '/work/groups/VEO/shared_data/bia_heyde/df_oxygen_' + feature + '_selected-filterNA.pickle.zst'  
#Output
#file2 = '/home/bia/Documents/BacterialData/oxygen/data/spearman_corr_df_oxygen_' + feature + '_selected-filterNA.pickle.zst'
file2 = '/work/no58rok/BacterialData/oxygen/data/spearman_corr_df_oxygen_' + feature + '_selected-filterNA.pickle.zst'
file3 = '/work/no58rok/BacterialData/oxygen/figures/spearman_corr_df_oxygen_' + feature + '_selected-filterNA.png' 
#file3 = '/home/bia/Documents/BacterialData/oxygen/figures/spearman_corr_df_oxygen_' + feature + '_selected-filterNA.png' 
file4 = '/work/no58rok/BacterialData/oxygen/figures/spearman_corr_df_oxygen_' + feature + '_selected-filterNA_0.10gap.png'
#file4 = '/home/bia/Documents/BacterialData/oxygen/figures/spearman_corr_df_oxygen_' + feature + '_selected-filterNA_0.10gap.png'
 
 
with zstandard.open(file1, 'rb') as f:
	df = pickle.load(f)

#Calculate Spearman correlation########################################

print(" Shape of the input dataframe:", df.shape)

# Separar as features (X) e o grupos (y)
#Full dataset:
X = df.iloc[:, :-1]  
y = df.iloc[:, -1] 

#Make plot title (plot is below)
plot_title = 'Correlation of prokaryotes\' ' + feature + ' (oxygen), n = ' + str(len(X.columns))

print("Calculating Spearman correlation...", datetime.datetime.now())

# Calcular a correlação de Spearman
correlation_matrix = X.corr(method='spearman')

print(" Shape of the correlation matrix:", correlation_matrix.shape)

print("Saving correlation dataframe...", datetime.datetime.now())

#Save calculations to file
with zstandard.open(file2, 'wb') as f:
	pickle.dump(correlation_matrix, f)
	
	

#Plot correlation histogram############################################


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

print("Plotting all values...", datetime.datetime.now())

# Plot the histogram full

plt.figure(figsize=(8, 6))
plt.hist(cleaned_list, bins=100, edgecolor='k', alpha=0.7);
plt.title(plot_title)
plt.xlabel('Sperman\'s rank correlation value');
plt.ylabel('Number of correlated pairs');
plt.savefig(file3, dpi=300)  


# Plot the histogram without values close to 0

print("Plotting values <= -0.1 or >= 0.1...", datetime.datetime.now())

filtered_list = [value for value in cleaned_list if value <= -0.1 or value >= 0.1]

plt.figure(figsize=(8, 6))
plt.hist(filtered_list, bins=100, edgecolor='k', alpha=0.7);
plt.title(plot_title)
plt.xlabel('Sperman\'s rank correlation value');
plt.ylabel('Number of correlated pairs');
plt.savefig(file4, dpi=300)  

print()
