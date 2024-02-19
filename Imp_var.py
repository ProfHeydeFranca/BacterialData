import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Carregar os dados
dados = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')

# Separar os dados em X e y
X = dados.drop(columns=['Salinity_group'])
y = dados['Salinity_group']

# Criar um modelo de floresta aleatória para selecionar características
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)

# Extrair a importância das características
feature_importances = rf_classifier.feature_importances_

# Criar um DataFrame para visualizar a importância das características
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Imprimir as 100 características mais importantes
print("As 100 características mais importantes:")
print(importance_df.head(100))

# Selecionar as características mais importantes
selected_features = SelectFromModel(rf_classifier, threshold='mean')
selected_features.fit(X, y)
X_selected = selected_features.transform(X)

# Verificar o número de características selecionadas
print("Número de características selecionadas:", X_selected.shape[1])
