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

"""
# Code in body
trials = Trials()
best_params = optimize(trials, space)


# Return the best parameters
space_eval(space, best_params)
"""
df = pd.read_csv("/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv", nrows="1000")
df.head()

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

# Complex Model
"""
 model = XGBRegressor(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=600,
    colsample_bytree=0.5,
    min_child_weight=330,
    subsample=0.8,    
    eta=0.2,
    objective='reg:squarederror',
    tree_method='gpu_hist')


model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=True, 
    early_stopping_rounds=10)
"""
#XGBoost
# Quick model
model = XGBRegressor(learning_rate = 0.05, n_estimators=300, max_depth=5)
model.fit(X_train, y_train)

# Predict the model
pred = model.predict(X_test)

# MAE Computation
scores_MAE = mean_absolute_error(y_test, pred)

# RMSE Computation
scores_RMSE = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE : % f, MAE : % f" % (scores_RMSE, scores_MAE)
"""
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores_MAE = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores_MAE = np.absolute(scores)

scores_RMSE = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores_RMSE = np.absolute(scores_RMSE)

print('Mean MAE: %.3f (%.3f)' % (scores_MAE.mean(), scores_MAE.std()))
print('Mean RMSE: %.3f (%.3f)' % (scores_RMSE.mean(), scores_RMSE.std()))
"""
"""
trials = Trials()
best_params = optimize(trials, space)
"""
"""
# Return the best parameters
space_eval(space, best_params)
"""
model = XGBRegressor(
    max_depth=5,
    colsample_bytree=0.6,
    n_estimators=300,
    min_child_weight=10,
    subsample=0.9,    
    eta=0.03,
    objective='reg:squarederror',
    #objective='reg:tweedie', tweedie_variance_power=1.54,
    #tree_method='gpu_hist'
    )


model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=False, 
    early_stopping_rounds=10)
"""
model = XGBRegressor(
    max_depth=9,
    colsample_bytree=0.6,
    n_estimators=300,
    min_child_weight=10,
    subsample=0.9,    
    eta=0.03,
    objective='reg:squarederror' #,
    #objective='reg:tweedie', tweedie_variance_power=1.53,
    #tree_method='gpu_hist'
)


model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=False, 
    early_stopping_rounds=10)
"""
# Predict the model
pred = model.predict(X_test)
 
# MAE Computation
scores_MAE = mean_absolute_error(y_test, pred)

# RMSE Computation
scores_RMSE = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE : % f, MAE : % f" % (scores_RMSE, scores_MAE))
"""
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores_MAE = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores_MAE = np.absolute(scores_MAE)
print('Mean MAE: %.3f (%.3f)' % (scores_MAE.mean(), scores_MAE.std()) )
"""
#Interpretation: Let's try graphing the predicted vs true values #!
#perm_predicted_vs_true = pd.merge(pd.DataFrame(pred_perm_subset),pd.DataFrame(y_perm_test))

predicted_vs_true = pd.DataFrame(y_test) 
#predictions = pd.DataFrame(pred_perm_subset)
predicted_vs_true['Predicted Salinity (XGBoost)'] = pred

#pd.concat([pd.DataFrame(pred_perm_subset),pd.DataFrame(y_perm_test)], axis=1, ignore_index=True)

#shap_value_sums
#y_df = pd.DataFrame(y)
#shap_valuecomp=pd.merge(y_df,shap_value_sums,left_index=True, right_index=True)
#shap_valuecomp
predicted_vs_true = predicted_vs_true.sort_index(ascending=True)
predicted_vs_true

#SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values_forFE = explainer.shap_values(X_test)
shap_values = explainer(X_test)

# create a 2D numpy array
arr = shap_values.values
# sum of each row
row_totals = arr.sum(axis=1)
row_totals = row_totals+shap_values[0].base_values
# display the array and the sum
print(arr)
#print("Sum of each row:", row_totals)
#row_totals.head()
shap_value_sums = pd.DataFrame(row_totals, columns = ['Shap_Prediction_Salinity'])
shap_value_sums = shap_value_sums.to_numpy()
y_df = pd.DataFrame(y_test)
shap_valuecomp = y_df
#print(shap_valuecomp.head())
shap_valuecomp['Predicted Salinity'] = shap_value_sums

shap_valuecomp = shap_valuecomp.sort_index(ascending=True)

#shap_valuecomp.plot(use_index=True)
plt.figure(figsize=(16, 9))
plt.plot(shap_valuecomp['Salinity group'], label = "line 2")
plt.plot(shap_valuecomp['Predicted Salinity'], label = "line 1")



plt.xlabel("Classes", fontsize = 15)
plt.ylabel("Salinity", fontsize = 15)
plt.suptitle("Optimal pH predictions vs XGBoost prediction on test set", fontsize=20)
plt.title("XGBoost without feature selection", fontsize = 15)

#plt.text(1, 2, r'an equation: $E=mc^2$', fontsize=15)
plt.text(1, 2, "RMSE: %.3f" % scores_RMSE, fontsize=15)
plt.text(1, 2.5, "MAE: %.3f" % scores_MAE, fontsize=15)

plt.savefig('/home/wi38kap/BacterialData/xgboost_testset.png')

plt.show()

explainer = shap.TreeExplainer(model)
shap_values_forFE = explainer.shap_values(X)
shap_values = explainer(X)



# create a 2D numpy array
arr = shap_values.values
# sum of each row
row_totals = arr.sum(axis=1)
row_totals = row_totals+shap_values[0].base_values
# display the array and the sum
#print(arr)
#print("Sum of each row:", row_totals)

shap_value_sums = pd.DataFrame(row_totals, columns = ['Shap_Salinity_prediction'])
shap_value_sums = shap_value_sums.to_numpy()
y_df = pd.DataFrame(y)
shap_valuecomp = y_df
shap_valuecomp['Predicted salinity (XGBoost)'] = shap_value_sums
print(shap_value_sums[2])
shap_valuecomp = shap_valuecomp.sort_index(ascending=True)
#shap_valuecomp = shap_valuecomp.sort_values(by='pH.preference', ascending=False).reset_index()
print(shap_valuecomp.head())
#shap_valuecomp.plot(use_index=True)
plt.figure(figsize=(16, 9))
plt.plot(shap_valuecomp['Salinity group'], label = "line 2")
plt.plot(shap_valuecomp['Predicted salinity (XGBoost)'], label = "line 1")


plt.xlabel("Class", fontsize = 15)
plt.ylabel("Salinity", fontsize = 15)
plt.suptitle("Optimal pH predictions vs SHAP explainer on full dataset", fontsize=20)
plt.title("BoostARoota-based feature selection", fontsize = 15)

#plt.text(1, 2, r'an equation: $E=mc^2$', fontsize=15)
plt.text(1, 2, "RMSE: %.3f" % scores_RMSE, fontsize=15)
plt.text(1, 2.5, "MAE: %.3f" % scores_MAE, fontsize=15)
fig.savefig('/home/wi38kap/BacterialData/xgboost_fullset.png')
plt.savefig('xgboost_fullset.png')
plt.show()
