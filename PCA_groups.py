import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar os dados
dados = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv',nrows=3)
#dados = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=3)
#dados = dados.type(float)
X = dados.select_dtypes(include=[float])

# Separar os dados em X e y
#X = dados.drop(columns=['Salinity_group','Unnamed: 0','Halophily', 'Class','Species', 'Best assembly_y'], axis=1)

# Iterar sobre as colunas e verificar o tipo de dados de cada uma
non_float_columns = []
for column in X.columns:
    if X[column].dtype != 'float64':
        non_float_columns.append(column)

# Imprimir as colunas que não contêm dados do tipo float
print("Colunas que não contêm dados do tipo float:")
for column_name in non_float_columns:
    print(column_name)
    
    
X = X.fillna(0)
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
#importancia_variaveis.to_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv')

# Plotar gráfico de importância das variáveis
plt.figure(figsize=(8, 6))
plt.plot(num_components_range, explained_variance_ratios, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada')
plt.title('Importância das Variáveis em relação ao Número de Componentes Principais')
plt.grid(True)
plt.savefig('/home/wi38kap/BacterialData/importancia_variaveis.png')  # Salvar o gráfico como imagem
#plt.savefig(r'C:\Users\00pau\importancia_variaveis.png')
plt.show()

# Salvar os resultados do PCA em um arquivo CSV
resultado_pca = pd.DataFrame(X_pca, columns=[f'Componente Principal {i+1}' for i in range(X_pca.shape[1])])
resultado_pca.to_csv('/home/wi38kap/BacterialData/resultado_pca.csv', index=False)
#resultado_pca.to_csv(r'C:\Users\00pau\resultado_pca.csv', index=False)

# Plotar gráfico de visualização dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Dados com PCA')
plt.colorbar(label='Salinity Group')
plt.grid(True)
plt.savefig('/home/wi38kap/BacterialData/visualizacao_dados_pca.png')  # Salvar o gráfico como imagem
#plt.savefig(r'C:\Users\00pau\visualizacao_dados_pca.png')

# Plotar matriz de covariância
covariance_matrix = np.cov(X_scaled.T)
plt.figure(figsize=(8, 6))
plt.imshow(covariance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Covariância')
plt.savefig('/home/wi38kap/BacterialData/matriz_covariancia.png')  # Salvar a matriz de covariância como imagem
#plt.savefig('r'C:\Users\00pau\matriz_covariancia.png') 
plt.show()
