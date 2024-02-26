from sklearn.preprocessing import LabelEncoder
from BorutaShap import BorutaShap, load_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=2)
df= pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')
df.head()
df = df.drop(['Unnamed: 0','Halophily', 'Class','Species'], axis=1)
df['Salinity group'] = df['Salinity group'].astype(str)
df['Salinity group'] = df['Salinity group'].fillna(0)

# If you want to replace NaN values with 0 in the entire DataFrame, you can use:
df.fillna(0, inplace=True)
#Split data into training features and labels

# Supondo que 'df' é o seu DataFrame e 'coluna' é o nome da coluna que contém as classes em forma de palavras
le = LabelEncoder()
df['Salinity group'] = le.fit_transform(df['Salinity group'])
X, y = df.loc[:, df.columns  != 'Salinity group'], df['Salinity group']
#X.head()
# Aplicar a codificação one-hot aos valores da variável de destino
# Converta todos os valores para string
#y_str = y.astype(str)


# Aplicar a codificação one-hot aos valores da variável de destino
#y_encoded = one_hot_encoder.fit_transform(y_str.values.reshape(-1, 1))
#y_df = pd.DataFrame(y_encoded.toarray())
print(y.head())

Feature_Selector = BorutaShap(importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=X, y=y, n_trials=100, sample=False,
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


# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
# Returns Boxplot of features and save it as an image
plot_path = "boruta_feature_selection_plot.png"
Feature_Selector.plot(which_features='all')
plt.savefig(plot_path)
plt.close()

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()

# Save the subset as a CSV file
subset_path = "selected_features_subset.csv"
subset.to_csv(subset_path, index=False)

print("Boxplot salvo em:", plot_path)
print("Subset de dados salvo em:", subset_path)