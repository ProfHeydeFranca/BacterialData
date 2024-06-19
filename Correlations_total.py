from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar o DataFrame
df = pd.read_csv('/home/wi38kap/BacterialData/df_filtered_total.csv')

# Preencher valores ausentes com 0
df.fillna(0, inplace=True)

# Separar as features (X) e o target (y)
X = df.iloc[:, :-1]  # Todas as colunas exceto a última
y = df.iloc[:, -1]  

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Calcular a correlação de Pearson
correlation_pearson = X.corr()
correlation_pearson.to_csv('correlation_pearson_total.csv')

# Calcular a correlação de Spearman
correlation_spearman = X.corr(method='spearman')
correlation_spearman.to_csv('correlation_spearman_total.csv')

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

# Criar e treinar o modelo de Random Forest
modelo = RandomForestRegressor()
modelo.fit(X_train, y_train)

importances = modelo.feature_importances_ 
indices = np.argsort(importances)
features = X.columns
plt.figure(figsize=(10,5))
plt.title('Most Important Features')
plt.barh(range(len(indices)), importances[indices], color='g', align='center',linestyle="solid",alpha=0.8)
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance')
plt.savefig('feature_importance_total.png')

# Shap para o modelo de Random Forest Regressor
explainer = shap.TreeExplainer(modelo, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.savefig('shap_random_forest_total.png')

# Salvar os plots do Shap em arquivos PNG
plt.savefig('shap_random_forest_total.png')
