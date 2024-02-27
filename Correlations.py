from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import shap
import numpy as np

df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=3, header=0)
#df= pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv', nrows=500, header=0)
df = df.drop(['Unnamed: 0','Halophily', 'Class','Species'], axis=1)
df['Salinity group'] = df['Salinity group'].astype(str)
df['Salinity group'] = df['Salinity group'].fillna(0)
df.fillna(0, inplace=True)

le = LabelEncoder()
df['Salinity group'] = le.fit_transform(df['Salinity group'])

target = 'Salinity group'
y = df[target]
X = df.drop(target, axis=1)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Calcula a correlação de Pearson
#correlation_pearson = df.corr()
correlation_pearson = X.iloc[:, :100].corr()
correlation_pearson.to_csv('correlation_pearson.csv')

# Calcula a correlação de Spearman
#correlation_spearman = df.corr(method='spearman')
correlation_spearman = X.iloc[:, :100].corr(method='spearman')
correlation_spearman.to_csv('correlation_spearman.csv')


# Inicializar o seletor
selector = SelectKBest(score_func=f_classif, k=10)  # Por exemplo, selecionar as 10 melhores características

# Aplicar o seletor aos dados
X_new = selector.fit_transform(X, y)

# Calcular as pontuações das características manualmente
feature_scores = -np.log10(selector.pvalues_)

# Classificar as características com base nas pontuações
sorted_indices = np.argsort(feature_scores)[::-1]
sorted_feature_scores = feature_scores[sorted_indices]

# Imprimir as 10 principais características e suas pontuações
for i in range(10):
    print(f"Feature {sorted_indices[i]}: Score {sorted_feature_scores[i]}")

#features = X_new.transform(x)
features = X_new
print(features)

# Criar e treinar o modelo de Random Forest
modelo = RandomForestRegressor()
modelo.fit(X_train, y_train)

importances = modelo.feature_importances_ 
indices = np.argsort(importances)
features = X.columns
plt.figure(figsize=(10,5))
plt.title('Features +importante')
plt.barh(range(len(indices)), importances[indices], color='g', align='center',linestyle="solid",alpha=0.8)
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importância')
plt.savefig('feature_importance.png')

#Shap para o modelo de Regressão Linear
lr = LinearRegression()
lr.fit(X_train, y_train)
explainer = shap.LinearExplainer(lr, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.savefig('shap_linear.png')

#Shap para o modelo de Random Forest Regressor
reg= RandomForestRegressor()
explainer = shap.TreeExplainer(reg, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.savefig('shap_random_forest.png')

explainer = shap.KernelExplainer(modelo.predict_proba,X[:100])
shap_values = explainer.shap_values(X[:100])
shap.summary_plot(shap_values, X[:100])
plt.savefig('shap_kernel.png')