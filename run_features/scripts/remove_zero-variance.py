#Importing packages
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


#Get feature from command line
if len(sys.argv) < 4:
    print("Usage: python script.py <feature> <abiotic_factor> <group>")
    sys.exit(1)

feature = sys.argv[1]
abiotic_factor = sys.argv[2]
pre_group = sys.argv[3]

#Replace underscore with space
group = pre_group.replace('_', ' ')

#REGRESSION #########################################
#CHANGE RANGE BELOW TO FIT GENE FAMILIES OR KMERS

#For different scoring of cross_validate, check: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter "Regression"
#For the output of cross_validate: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

#Benchmark different thresholds of filtering low-variance features

f1_df = pd.DataFrame()

#path = '/home/bia/Documents/BacterialData/run_features/benchmark_low-variance_threshold/df_'
path = '/vast/no58rok/BacterialData/run_features/benchmark_low-variance_threshold/df_'

#Loop for different thresholds for filtering low-variance
#for i in np.arange(0, 0.011, 0.001):  
for i in np.arange(0, 0.0021, 0.0002):  
        
    with zstandard.open(path + abiotic_factor + '_' + 
                        feature + '_' + str(i) + '.pickle.zst', 'rb') as f: 
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
#    for it in range(1, 3, 1):        

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
    f1_df.to_csv(path + 'mae_benchmark_' + abiotic_factor + '_' + feature + '.csv', index=True)

#Save benchmark results to a CSV file
f1_df.to_csv(path + 'mae_benchmark_' + abiotic_factor + '_' + feature + '.csv', index=True)