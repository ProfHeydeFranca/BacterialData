# Data wrangling
import pandas as pd

# Scientific
import numpy as np

# Hyperparameters tuning
try:
    from hpsklearn import HyperoptEstimator, any_classifier
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
except:
     #!pip install hpsklearn
    from hpsklearn import HyperoptEstimator, any_classifier
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# Machine learning
try:
    #import xgboost as xgb
    from xgboost import XGBRegressor
except:
     #!pip install xgboost
    from xgboost import XGBRegressor
    #import xgboost as xgb

    
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn import preprocessing

try:
    from boruta import BorutaPy
except:
     #!pip install boruta
    from boruta import BorutaPy

try:
    from boostaroota import BoostARoota
except:
     #!pip install boostaroota
    from boostaroota import BoostARoota
    
import pickle
    
try:
    import shap
except:
     #!pip install shap
    import shap
    
# Graphics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns # for correlation heatmap
from boruta import BorutaPy
seed = 100
# Choose hyperparameter domain to search over
space = {
        'max_depth':hp.choice('max_depth', np.arange(4, 25, 1, dtype=int)),
        'n_estimators':hp.choice('n_estimators', np.arange(100, 10000, 10, dtype=int)),
        'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
        'min_child_weight':hp.choice('min_child_weight', np.arange(250, 350, 10, dtype=int)),
        'subsample':hp.quniform('subsample', 0.7, 0.9, 0.1),
        'eta':hp.quniform('eta', 0.01, 0.15, 0.01),
        'learning_rate':hp.quniform('learning_rate', 0.01, 0.05, 0.01),
        'objective':'reg:squarederror',
        'tree_method':'gpu_hist',
        'eval_metric': 'rmse',
    }


def score(params):
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False, early_stopping_rounds=10)
    y_pred = model.predict(X_test).clip(0, 20)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    print(score)
    return {'loss': score, 'status': STATUS_OK}    
    

def optimize(trials, space):
    best = fmin(score, space, algo=tpe.suggest, max_evals=1000)
    return best


model = XGBRegressor(
    max_depth=17,
    n_estimators=2110,
    colsample_bytree=0.5,
    min_child_weight=330,
    subsample=0.8,    
    eta=0.2,
    objective='reg:squarederror',
    tree_method='gpu_hist')

#df = pd.read_csv("/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv", nrows="2")
df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv',nrows=2)
print(df.head())

#Here we are going to get rid of columns we don't need
AVS_list = df['Unnamed: 0']
group_list= df['Salinity group']
df = df.drop(['Unnamed: 0','Halophily', 'Class','Species'], axis=1)
df.head()
AVS_list.head()
group_list.head()

# REmover os nan da coluna de SAlinity e substituir por 0
df['Salinity group'] = df['Salinity group'].fillna(0)

# If you want to replace NaN values with 0 in the entire DataFrame, you can use:
df.fillna(0, inplace=True)
gene_list = list(df)
gene_list.pop(-1)


#Now that our data is loaded and trimed down to its potentially useful elements, it's time to prep it for machine learning algorithms
#Split data into training features and labels
X, y = df.loc[:, df.columns  != 'Salinity group'], df['Salinity group']
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# define Boruta feature selection method
feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features
feat_selector.fit(X_train.values, y_train.values)

# check selected features
feat_selector.support_
accept = X.columns[feat_selector.support_].to_list()

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X_train.values)
# zip my names, ranks, and decisions in a single iterable
feature_ranks = list(zip(list(X_train.columns), 
                         feat_selector.ranking_, 
                         feat_selector.support_))

# iterate through and print out the results
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))

accept = X.columns[feat_selector.support_].to_list()
accept.insert(0, "pH.preference")
accept

new_x = df[accept]

X_new, y_new = new_x.loc[:, new_x.columns != 'pH.preference'], df['pH.preference']

X_t, X_val, y_t, y_val = train_test_split(X_new, y_new, random_state=42)



model.fit(
    X_t, 
    y_t, 
    eval_metric="rmse", 
    eval_set=[(X_t, y_t), (X_val, y_val)], 
    verbose=True, 
    early_stopping_rounds=10)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )
