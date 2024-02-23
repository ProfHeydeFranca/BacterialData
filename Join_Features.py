import pandas as pd

# Carregar os dados
#dados = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv', nrows=3)
dados = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=3)
#transformar dados categóricos
print(dados.head())

# Drop the collumns with categorical data 

#dados['Halophily'] = dados['Halophily'].astype('category')
#dados['Class'] = dados['Class'].astype('category')
#dados['Species'] = dados['Species'].astype('category')
dados = dados.drop(['Unnamed: 0', 'Halophily', 'Class', 'Species', 'Best assembly_y'], axis=1)
#dados['Salinity group'] = dados['Salinity group'].fillna(0)
dados = dados.fillna(0)
# Lista de colunas que contêm dados de texto

#cols_with_text_data = ['Halophily', 'Class', 'Species', 'Best assembly_y']

# Criar DataFrame codificado para variáveis categóricas
#dados_encoded = pd.get_dummies(dados[cols_with_text_data])

# Remover as colunas originais do DataFrame 'dados'
#dados = dados.drop(columns=cols_with_text_data)

# Concatenar os DataFrames 'dados' e 'dados_encoded'
#dados = pd.concat([dados, dados_encoded], axis=1)

# Iterar sobre as colunas e verificar o tipo de dados de cada uma
non_float_columns = []
for column in dados.columns:
    if dados[column].dtype != 'float64':
        non_float_columns.append(column)

# Imprimir as colunas que não contêm dados do tipo float
print("Colunas que não contêm dados do tipo float:")
for column_name in non_float_columns:
    print(column_name)
    
# Calcular a matriz de correlação
correlation_matrix = dados.corr()

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
        # Adicionar outras colunas ao grupo e marcar como já agrupadas
        for correlated_col in correlated_columns:
            column_groups[correlated_col] = correlated_columns

# Criar um novo DataFrame para armazenar as colunas agrupadas
dados_agrupados = pd.DataFrame()

# Iterar sobre os grupos de colunas e calcular estatísticas resumidas
for group, columns in column_groups.items():
    # Calcular estatísticas resumidas para as colunas no grupo
    group_data = dados[columns].agg(['mean', 'median', 'std']).T
    # Adicionar o nome do grupo como prefixo para as estatísticas resumidas
    group_data.columns = [f'{group}_{stat}' for stat in group_data.columns]
    # Adicionar as estatísticas resumidas ao DataFrame agrupado
    dados_agrupados = pd.concat([dados_agrupados, group_data], axis=1)

# Salvar o DataFrame agrupado em um arquivo CSV
dados_agrupados.to_csv('/home/wi38kap/BacterialData/dados_agrupados.csv', index=False)