import xgboost
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler        
from sklearn.decomposition import PCA 
from sklearn.preprocessing import LabelEncoder
from BorutaShap import BorutaShap, load_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=8, header=0)
#df= pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')
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

target = ['Salinity group']
y = df[target]
print(df[target])
X = df.loc[:, df.columns  != 'Salinity group']
print(y)
print(target)
print(X)
#Scale Etmemiz gerekiyor;

x =StandardScaler().fit_transform(X)
print(x)#scale durumuna getirdik.


# train an XGBoost model

model = xgboost.XGBRegressor().fit(x, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(x)


# Convertendo os valores SHAP para um DataFrame do pandas
shap_df = pd.DataFrame(shap_values.values, columns=X.columns)

# Salvando o DataFrame em um arquivo CSV
shap_df.to_csv('shap_values.csv', index=False)

# visualize the first prediction's explanation
plt.savefig('shap.png', dpi=300)
shap.plots.waterfall(shap_values[0])
  
#plot 
plt.savefig('shap2v.png', dpi=300)
shap.summary_plot(shap_values, X, plot_type="bar")

#plot 
plt.savefig('shap3v.png', dpi=300)
shap.summary_plot(shap_values, X, plot_type="violin")