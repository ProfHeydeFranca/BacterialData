import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar os dados
dados = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')

# Separar os dados em X e y
X = dados.drop(columns=['Salinity_group'])
y = dados['Salinity_group']

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar um grid para testar diferentes números de componentes principais
num_components_range = np.arange(1, 1001, 50)

# Lista para armazenar a variância explicada
explained_variance_ratios = []

# Loop sobre diferentes números de componentes principais
for n_components in num_components_range:
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

# Salvar importância das variáveis em um arquivo CSV
importancia_variaveis = pd.DataFrame(explained_variance_ratios, columns=['Importancia'])
importancia_variaveis.to_csv('/home/wi38kap/BacterialData/importancia_variaveis.csv', index=False)

# Plotar gráfico de importância das variáveis
plt.figure(figsize=(8, 6))
plt.plot(num_components_range, explained_variance_ratios, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada')
plt.title('Importância das Variáveis em relação ao Número de Componentes Principais')
plt.grid(True)
plt.savefig('/home/wi38kap/BacterialData/importancia_variaveis.png')  # Salvar o gráfico como imagem
plt.show()

# Salvar os resultados do PCA em um arquivo CSV
resultado_pca = pd.DataFrame(X_pca, columns=[f'Componente Principal {i+1}' for i in range(X_pca.shape[1])])
resultado_pca.to_csv('/home/wi38kap/BacterialData/resultado_pca.csv', index=False)

# Plotar gráfico de visualização dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Dados com PCA')
plt.colorbar(label='Salinity Group')
plt.grid(True)
plt.savefig('/home/wi38kap/BacterialData/visualizacao_dados_pca.png')  # Salvar o gráfico como imagem
plt.show()


# Plotar gráfico de visualização dos dados com cores diferentes para cada classe de salinidade
plt.figure(figsize=(8, 6))
for salinity_group in ['Low', 'Medium', 'High']:
    plt.scatter(X_pca[y == salinity_group, 0], X_pca[y == salinity_group, 1], label=salinity_group)

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Dados com PCA')
plt.legend()
plt.grid(True)
plt.savefig('/home/wi38kap/BacterialData/visualizacao_dados_pca_colorido.png')  # Salvar o gráfico como imagem
plt.show()

# Plotar matriz de covariância
covariance_matrix = np.cov(X_scaled.T)
plt.figure(figsize=(8, 6))
plt.imshow(covariance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Covariância')
plt.savefig('/home/wi38kap/BacterialData/matriz_covariancia.png')  # Salvar a matriz de covariância como imagem
plt.show()
