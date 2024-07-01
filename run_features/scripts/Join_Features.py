#After calculating correlations between genomic features, this script groups features that have a correlation higher than 0.9
#It outputs a df_salt/df_oxygen dataframe such as the one below
#From feature_selection.ipynb

#Best assembly	COG0606@2	33SZW@2,33UZ0@2,2ZFEM@2,33E1D@2,34282@2,33QA2@2,COG2385@1,32DYM@2	COG0774@1	Target																			
#1002367.3	0.00	1.00	1.00	anaerobe
#108980.91	0.00	0.00	0.00	aerobe
#1111140.3	1.00	0.00	0.00	aerobe

import sys
import pickle
import zstandard
import datetime
import pandas as pd
import numpy as np

#Get feature from command line
if len(sys.argv) < 2:
    print("Usage: python script.py <feature>")
    sys.exit(1)

feature = sys.argv[1]

print()

print("Started script! Loading input files...", datetime.datetime.now())

#Input
file1 = '/work/groups/VEO/shared_data/bia_heyde/df_oxygen_' + feature + '_selected-filterNA.pickle.zst'
#file1 = '/home/bia/Documents/bacterial_phenotypes/connecting_features_abFactors/df_oxygen_' + feature + '_selected-filterNA.pickle.zst'
file2 = '/work/no58rok/BacterialData/oxygen/data/spearman_corr_df_oxygen_' + feature + '_selected-filterNA.pickle.zst'
#file2 = '/home/bia/Documents/BacterialData/oxygen/data/spearman_corr_df_oxygen_' + feature +  '_selected-filterNA.pickle.zst'
#Output
file3 = '/work/no58rok/BacterialData/oxygen/data/Oxygen_' + feature +  '_Features_Corr_Spearman_with_Target.pickle.zst'
#file3 = '/home/bia/Documents/BacterialData/oxygen/data/Oxygen_' + feature +  '_Features_Corr_Spearman_with_Target.pickle.zst'

with zstandard.open(file1, 'rb') as f:
#with zstandard.open(file2, 'rb') as f:
	df = pickle.load(f)

with zstandard.open(file2, 'rb') as f:
	correlation_matrix = pickle.load(f)

print(" Shape of the input dataframe df_oxygen:", df.shape)
print(" Shape of the input dataframe spearman_corr_df:", correlation_matrix.shape)


#Join highly correlated features#######################################	

print("Identifying highly correlated columns...", datetime.datetime.now())

# Definir o limite de valor de correlação para agrupamento
threshold = 0.9

# Inicializar um dicionário para armazenar os grupos de colunas
column_groups = {}

# Iterar sobre as colunas e identificar grupos de colunas altamente correlacionadas
for col in correlation_matrix.columns:

    #print(col)
    
    # Verificar se a coluna já foi agrupada
    if col not in column_groups:
        
        # Encontrar colunas altamente correlacionadas com a coluna atual
        correlated_columns = correlation_matrix.index[correlation_matrix[col] > threshold].tolist()

        #print(correlated_columns)
        
        # Adicionar a coluna atual ao grupo
        column_groups[col] = correlated_columns
        
        # Marcar outras colunas do grupo como já agrupadas
        for correlated_col in correlated_columns:
            if correlated_col != col:
                column_groups[correlated_col] = correlated_columns
                
print("Filtering unique groups of columns...", datetime.datetime.now())

# Filtrar grupos únicos de colunas
unique_groups = []
already_seen = set()

for group in column_groups.values():
    
    group_tuple = tuple(sorted(group))

    if group_tuple not in already_seen:
        unique_groups.append(group)
        already_seen.add(group_tuple)
        
print("Storing grouped columns...", datetime.datetime.now())

# Criar um novo DataFrame para armazenar as colunas agrupadas
dados_agrupados = pd.DataFrame()

# Iterar sobre os grupos de colunas e calcular estatísticas resumidas
for group in unique_groups:
    # Usar a média das colunas no grupo como representação
    group_mean = df[group].mean(axis=1)

    group_str = ','.join(group)
    pre_group_name = group_str.replace("[", "").replace("]", "").replace("'", "")
    group_name = f'{pre_group_name}'
    #group_name = f'Group_{group[0]}'

    dados_agrupados[group_name] = group_mean
    
print("Adding target column...", datetime.datetime.now())

# Adicionar a coluna y ao DataFrame agrupado
y = df.iloc[:, -1]  # Última coluna

#Get shape without target!
print(" Shape of the output dataframe dados_agrupados_df:", dados_agrupados.shape)

dados_agrupados['Target'] = y

print("Saving file...", file3, datetime.datetime.now())

with zstandard.open(file3, 'wb') as f:
	pickle.dump(dados_agrupados, f)
      
print("Finished script", datetime.datetime.now())
print()        
