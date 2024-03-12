import matplotlib.pyplot as plt
import pandas as pd
import os

# Função para plotar os gráficos
def plotar_graficos(dados):
    plt.figure(figsize=(12, 8))

    # Loop sobre os modelos
    for modelo, df in dados.items():
        # Verificando se as colunas necessárias estão presentes no DataFrame
        colunas_necessarias = ['Acuracy', 'Precision', 'Recall', 'F-Score']
        if all(coluna in df.columns for coluna in colunas_necessarias):
            modelos = df.index.tolist()  # Usando o índice do DataFrame como modelos
            for metrica in colunas_necessarias:
                plt.plot(modelos, df[metrica], marker='o', label=f'{modelo} - {metrica}')
            plt.title(f'Métricas - {modelo}')
            plt.xlabel('Dataset')
            plt.ylabel('Valor')
            plt.legend()
            plt.show()
        else:
            print(f"O DataFrame para o modelo {modelo} não contém todas as colunas necessárias.")

# Diretório onde os arquivos CSV estão localizados
diretorio = 'G:/Meu Drive/ProjetoAlemanha/Testes/dados_plot'  # Substitua pelo seu diretório

# Lista para armazenar os dados de cada arquivo CSV
dados = {}

# Iterar sobre os arquivos CSV no diretório
for arquivo in os.listdir(diretorio):
    if arquivo.endswith('.csv'):
        nome_modelo = os.path.splitext(arquivo)[0]  # Nome do modelo será o nome do arquivo sem a extensão
        caminho_arquivo = os.path.join(diretorio, arquivo)
        df = pd.read_csv(caminho_arquivo, sep=';', index_col='Dataset')  # Usando 'Dataset' como índice
        dados[nome_modelo] = df
        print(f"Dados do modelo {nome_modelo}:")
        print(dados[nome_modelo])

# Plotar os gráficos
plotar_graficos(dados)
