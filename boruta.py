from BorutaShap import BorutaShap, load_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=3)
df.head()
df = df.drop(['Unnamed: 0','Halophily', 'Class','Species'], axis=1)
df['Salinity group'] = df['Salinity group'].fillna(0)

# If you want to replace NaN values with 0 in the entire DataFrame, you can use:
df.fillna(0, inplace=True)
#Split data into training features and labels

# Instanciar o OneHotEncoder
one_hot_encoder = OneHotEncoder()

X, y = df.loc[:, df.columns  != 'Salinity group'], df['Salinity group']
#X.head()
# Aplicar a codificação one-hot aos valores da variável de destino
# Converta todos os valores para string
y_str = y.astype(str)

# Instanciar o OneHotEncoder
one_hot_encoder = OneHotEncoder()

# Aplicar a codificação one-hot aos valores da variável de destino
y_encoded = one_hot_encoder.fit_transform(y_str.values.reshape(-1, 1))
y_df = pd.DataFrame(y_encoded.toarray())

Feature_Selector = BorutaShap(importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=X, y=y_df, n_trials=100, sample=False,
            	     train_or_test = 'test', normalize=True,
		     verbose=True)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all')

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()

from BorutaShap import BorutaShap, load_data
from xgboost import XGBClassifier

X, y = load_data(data_type='classification')
X.head()

model = XGBClassifier()

# if classification is False it is a Regression problem
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=True)

Feature_Selector.fit(X=X, y=y_encoded, n_trials=100, sample=False,
            	     train_or_test = 'test', normalize=True,
		     verbose=True)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all')


# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()