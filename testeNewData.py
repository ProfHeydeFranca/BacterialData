import pandas as pd

# Carregar dados do primeiro arquivo CSV (bactérias)
df= pd.read_csv('/work/no58rok/bacterial_phenotypes_draco/df_salt_objects/df_salt_filtered-salinity_best_assembly_temp_pH_oxygen_merged_assemblies.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)


print('DataFrame')
#print(df)
print(df.dtypes)

df= df.drop('Unnamed: 0', axis=1)
print(df.dtypes)

# Converter variáveis categóricas em numéricas usando LabelEncoder
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# Converter variáveis categóricas restantes em variáveis dummy (one-hot encoding)
dados_bacterias = pd.get_dummies(df)

# Criar lista de dados
lista_de_dados = []

print("Read df_salt")

dados_genomas = pd.read_csv('/work/no58rok/bacterial_phenotypes_draco/df_features_objects/df_features_halophily_pipeline2.2_merged.csv')


print("Read df_features")
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
