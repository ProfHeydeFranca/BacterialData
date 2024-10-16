#REGRESSION #########################################

#Benchmark different thresholds of joining highly correlated features
#Based on: Benchmark of "remove_zero-variance.ipynb/py"

#Importing packages
import datetime
import pickle
import zstandard
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys

#Cross-validation packages
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import statistics
from sklearn.ensemble import RandomForestRegressor

print()
print("Started script! Loading input file...", datetime.datetime.now())

#Define variables
#feature = 'kmer9Classification'
feature = 'kmer9Regression'
#feature = 'gene-familiesClassification'
#feature = 'gene-familiesRegression'
#feature = 'gene-families'
#feature = 'kmer9'

abiotic_factor = 'salt'
#abiotic_factor = 'temperature'
#abiotic_factor = 'pH'
#abiotic_factor = 'oxygen'

#Define target group:
#group = 'Salinity group'
group = 'Salt all mean'

#var_filter = '0.009000000000000001final-filtering' #salt Class GF
#var_filter = '0.001' #salt Reg GF
#var_filter = '0.001final-filtering' #salt Class kmer9
var_filter = '0.0014' #salt Reg kmer9

#Input
path = '../benchmark_low-variance_threshold/'
file = path + 'df_' + abiotic_factor + '_' + feature + '_' + var_filter + '.pickle.zst'  

path_out = '../join_highly_correlated/'
file_out_filtered = path_out + 'df_' + abiotic_factor + '_' + feature  


#REGRESSION #########################################

#For different scoring of cross_validate, check: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter "Regression"
#For the output of cross_validate: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

f1_df = pd.DataFrame()

print("Started script on: ", datetime.datetime.now())

#Loop for different thresholds for features joining
for i in np.arange(0.65, 0.99, 0.05): 

    if(i == 0.65):
        i = 'no filter'
        with zstandard.open(file, 'rb') as f:
        	tmp = pickle.load(f)
    elif(i == 0.7000000000000001):
        i = 0.7
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    elif(i == 0.7500000000000001):
        i = 0.75
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    elif(i == 0.8000000000000002):
        i = 0.80
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    elif(i == 0.8500000000000002):
        i = 0.8500000000000001
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    elif(i == 0.9000000000000002):
        i = 0.9000000000000001
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    elif(i == 0.9500000000000003):
        i = 0.9500000000000002
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    else:
        with zstandard.open(file_out_filtered + '_' + str(i) + '.pickle.zst', 'rb') as f:                         
            tmp = pickle.load(f)
    
    print('Calculating metrics for threshold', i, 'of filtering low-variance features...')
    print('  Shape of input data:', tmp.shape)
    
    list_means = []

    #Separating features from group/target variable
    X = tmp.drop(group, axis=1)
    y = tmp[group]

    #I included random_state to make this command reproducible for after feature selection
    #Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)

    print('  Shape of training data:', X_train.shape)
    
    #Do 10 iterations for every different threshold
    for it in range(1, 11, 1):   

        #Cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=it)
        #Instantiate Random Forest model
        clf = RandomForestRegressor()

        scoring = 'neg_mean_absolute_error'
        output = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_train_score=False, return_estimator=True, n_jobs=15)
        #print(output)
        #{'fit_time': array([57.96, 62.67, 56.21, 67.85, 60.65]), 'score_time': array([0.05, 0.05, 0.05, 0.05, 0.05]), 
        #'estimator': [RandomForestRegressor(), RandomForestRegressor(), RandomForestRegressor(), RandomForestRegressor(), RandomForestRegressor()], 
        #'test_score': array([-13.17, -12.15, -17.07, -19.26, -32.13])}

        #Get mean MAE for this iteration
        mean = statistics.mean(output['test_score'])
        #Add mean MAE to vector containing results of all iterations
        list_means.append(mean)

        print(' Iteration', it, 'has as mean absolute error =', round(mean, 3), 'All errors:', output['test_score'])
    
    print(' Mean absolute error of cross-validation for all iterations:', round(statistics.mean(list_means), 3))

    #Add new values
    f1_df[i] = list_means
    
    #Save benchmark results to a CSV file
    f1_df.to_csv(file_out_filtered + '_' + 'MAE_benchmarking.csv', index=True)

#Save benchmark results to a CSV file
f1_df.to_csv(file_out_filtered + '_' + 'MAE_benchmarking.csv', index=True)