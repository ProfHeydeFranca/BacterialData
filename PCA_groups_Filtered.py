import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar os dados
df = pd.read_csv('/home/wi38kap/BacterialData/df_filtered.csv')
#df = pd.read_csv(r"C:\Users\00pau\df_filtered.csv", nrows=600)
#dados = dados.type(float)
X = df.iloc[:, :-1]  # Todas as colunas exceto a última
y = df.iloc[:, -1]   # Última coluna
X = X.select_dtypes(include=[float])
X= X.fillna(0)
y= y.fillna(0)


# Mapear as classes para valores numéricos
class_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_mapped = y.map(class_mapping)
print(y_mapped.head(2))
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
importancia_variaveis.to_csv('/home/wi38kap/BacterialData/importancia_variaveis_filtered.csv', index=False)
#importancia_variaveis.to_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv')

# Plotar gráfico de importância das variáveis
plt.figure(figsize=(8, 6))
plt.plot(num_components_range, explained_variance_ratios, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada')
plt.title('Importância das Variáveis em relação ao Número de Componentes Principais')
plt.grid(True)
plt.savefig('/home/wi38kap/BacterialData/importancia_variaveis_filtered.png')  # Salvar o gráfico como imagem
#plt.savefig(r'C:\Users\00pau\importancia_variaveis.png')
plt.show()

# Salvar os resultados do PCA em um arquivo CSV
resultado_pca = pd.DataFrame(X_pca, columns=[f'Componente Principal {i+1}' for i in range(X_pca.shape[1])])
resultado_pca.to_csv('/home/wi38kap/BacterialData/resultado_pcafiltered.csv', index=False)
#resultado_pca.to_csv(r'C:\Users\00pau\resultado_pca.csv', index=False)

# Plotar gráfico de visualização dos dados
# Mapear cores para as classes 'low', 'medium' e 'high'
colors = {0: 'blue', 1: 'green', 2: 'red'}

# Plotar o gráfico de dispersão com as cores mapeadas
plt.figure(figsize=(10, 6))
for i in range(len(X_pca)):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[y_mapped[i]], alpha=0.5)
plt.title("Gráfico de Dispersão (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.savefig('scatter_plot_PCA.png')
plt.show()


# Plotar matriz de covariância
covariance_matrix = np.cov(X_scaled.T)
plt.figure(figsize=(8, 6))
plt.imshow(covariance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Covariância')
plt.savefig('/home/wi38kap/BacterialData/matriz_covarianciafiltered.png')  # Salvar a matriz de covariância como imagem
#plt.savefig(r'C:\Users\00pau\matriz_covariancia.png') 
plt.show()