import matplotlib.pyplot as plt
import pandas as pd
import os

# Função para plotar os gráficos de barras
def plotar_graficos(dados):
    plt.figure(figsize=(12, 8))

    # Loop sobre os modelos
    for modelo, df in dados.items():
        # Verificando se as colunas necessárias estão presentes no DataFrame
        colunas_necessarias = ['Acuracy', 'Precision', 'Recall', 'F-Score']
        if all(coluna in df.columns for coluna in colunas_necessarias):
            modelos = df.index.tolist()  # Usando o índice do DataFrame como modelos
            num_metricas = len(colunas_necessarias)

            # Configuração do subplot
            plt.subplot(1, len(dados), list(dados.keys()).index(modelo) + 1)

            # Espaçamento entre as barras
            width = 0.2

            # Posições das barras para cada métrica
            positions = range(len(modelos))

            # Plotagem das barras para cada métrica
            for i, metrica in enumerate(colunas_necessarias):
                plt.bar([p + width*i for p in positions], df[metrica], width=width, label=metrica)

            plt.title(f'Métricas - {modelo}')
            plt.xlabel('Dataset')
            plt.ylabel('Valor')
            plt.legend()
            plt.xticks([p + 1.5 * width for p in positions], modelos, rotation=45)
            plt.ylim(0, 1)  # Definindo os limites do eixo y
            plt.tight_layout()

    plt.show()

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

# Plotar os gráficos de barras
plotar_graficos(dados)
