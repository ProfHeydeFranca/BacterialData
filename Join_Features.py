import pandas as pd

# Carregar os dados
dados = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')

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
