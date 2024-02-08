import pandas as pd
from tqdm import tqdm  # Importar tqdm para a barra de progresso

# Ler apenas a primeira linha do arquivo CSV
primeira_linha = pd.read_csv('G:/Meu Drive/ProjetoAlemanha/Testes/df_features_halophily_pipeline2.2_merged.csv', nrows=2)

# Imprimir a primeira linha
print(primeira_linha)

######
# Carregar dados do primeiro arquivo CSV (bactérias)
dados_bacterias = pd.read_csv('G:/Meu Drive/ProjetoAlemanha/Testes/df_salt_filtered-salinity_best_assembly - df_salt_filtered-salinity_best_assembly.csv')
print(dados_bacterias.columns)
# Criar lista de dados
lista_de_dados = []

# Definir tamanho do bloco para leitura
chunksize = 1000

# Ler o arquivo CSV em blocos menores
for dados_genomas_chunk in pd.read_csv('G:/Meu Drive/ProjetoAlemanha/Testes/df_features_halophily_pipeline2.2_merged.csv', chunksize=chunksize):
    # Iterar sobre as linhas do bloco de dados do segundo arquivo
    for indice, linha in dados_genomas_chunk.iterrows():
        # Verificar se 'Best assembly' coincide com algum valor do primeiro arquivo
        if linha['Unnamed: 0'] in dados_bacterias['Best assembly'].values:
            # Obter a linha correspondente do primeiro arquivo
            linha_bacteria = dados_bacterias[dados_bacterias['Best assembly'] == linha['Best assembly']].iloc[0]

            # Copiar todas as colunas do segundo arquivo
            dados_completos = linha.to_dict()

            # Adicionar colunas específicas do primeiro arquivo
            dados_completos.update({
                'Class': linha_bacteria['Class'],
                'Species': linha_bacteria['Species'],
                'Salt optimum': linha_bacteria['Salt optimum'],
                'Salt all': linha_bacteria['Salt all']
            })

            # Adicionar dados completos à lista de dados
            lista_de_dados.append(dados_completos)

# Converter lista de dados para DataFrame
df_dados_final = pd.DataFrame(lista_de_dados)

# Salvar o DataFrame em um novo arquivo CSV
df_dados_final.to_csv('dados_bacterias_com_genomas.csv', index=False)

print("Dados salvos com sucesso!")
