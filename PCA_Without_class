import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar os dados
dados = pd.read_csv('/work/groups/VEO/shared_data/bia_heyde/df_features_halophily_pipeline2.2_merged.csv')

# Atribuir todos os dados do arquivo CSV à variável X
X = dados.values

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variância explicada por cada componente
explained_variance_ratio = pca.explained_variance_ratio_

# Salvar importância das variáveis em um arquivo CSV
importancia_variaveis = pd.DataFrame(explained_variance_ratio, columns=['Importancia'])
importancia_variaveis.to_csv('importancia_variaveis.csv', index=False)

# Salvar os resultados do PCA em um arquivo CSV
resultado_pca = pd.DataFrame(X_pca, columns=[f'Componente Principal {i+1}' for i in range(X_pca.shape[1])])
resultado_pca.to_csv('resultado_pca.csv', index=False)

# Plotar gráfico de visualização dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Dados com PCA')
plt.colorbar(label='Classe')
plt.grid(True)
plt.savefig('visualizacao_dados_pca.png')  # Salvar o gráfico como imagem
plt.show()

# Plotar matriz de covariância
covariance_matrix = np.cov(X_scaled.T)
plt.figure(figsize=(8, 6))
plt.imshow(covariance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Covariância')
plt.savefig('matriz_covariancia.png')  # Salvar a matriz de covariância como imagem
plt.show()
