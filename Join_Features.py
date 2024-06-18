import pandas as pd
import numpy as np

# Carregar os dados
file_path = r'/home/wi38kap/BacterialData/df_filtered.pickle'

# Ler o arquivo .pickle em um DataFrame
df = pd.read_pickle(file_path)

# Preencher valores faltantes
df = df.fillna(0)

# Imprimir as duas primeiras linhas do DataFrame
print("As duas primeiras linhas do DataFrame:")
print(df.head(2))

# Imprimir o formato do DataFrame
print("\nFormato do DataFrame (número de linhas, número de colunas):")
print(df.shape)

# Imprimir os tipos de dados das colunas do DataFrame
print("\nTipos de dados das colunas:")
print(df.dtypes)

X = df.iloc[:, :-1] # All columns except the last one
y = df.iloc[:, -1]   # Last column

# Map classes to numeric values
class_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_mapped = y.map(class_mapping)
print(y_mapped.head(2))



# Calcular a matriz de correlação para colunas numéricas
correlation_matrix = X.corr()

# Definir o limite de correlação para agrupamento
threshold = 0.9

# Inicializar um dicionário para armazenar os grupos de colunas
column_groups = {}

# Iterar sobre as colunas e identificar grupos de colunas altamente correlacionadas
for col in correlation_matrix.columns:
    # Verificar se a coluna já foi agrupada
    if col not in column_groups:
        # Encontrar colunas altamente correlacionadas com a coluna atual
        correlated_columns = correlation_matrix.index[correlation_matrix[col] > threshold].tolist()
        # Adicionar a coluna atual ao grupo
        column_groups[col] = correlated_columns
        # Marcar outras colunas do grupo como já agrupadas
        for correlated_col in correlated_columns:
            if correlated_col != col:
                column_groups[correlated_col] = correlated_columns

# Filtrar grupos únicos de colunas
unique_groups = []
already_seen = set()
for group in column_groups.values():
    group_tuple = tuple(sorted(group))
    if group_tuple not in already_seen:
        unique_groups.append(group)
        already_seen.add(group_tuple)

# Criar um novo DataFrame para armazenar as colunas agrupadas
dados_agrupados = pd.DataFrame()

# Iterar sobre os grupos de colunas e calcular estatísticas resumidas
for group in unique_groups:
    # Usar a média das colunas no grupo como representação
    group_mean = df[group].mean(axis=1)
    group_name = f'Group_{group[0]}'
    dados_agrupados[group_name] = group_mean

# Adicionar a coluna y ao DataFrame agrupado
y = df.iloc[:, -1]  # Última coluna
dados_agrupados['Target'] = y

# Salvar o DataFrame agrupado em um arquivo pickle
output_file_path = r'/home/wi38kap/BacterialData/Features_Corr_with_Target.pickle'
dados_agrupados.to_pickle(output_file_path)

# Verificação para garantir que o arquivo foi salvo corretamente
print("DataFrame salvo como arquivo .pickle.")
