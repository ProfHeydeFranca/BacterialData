import pandas as pd
#Reading the data
df_features = pd.read_csv('work/groups/VEO/shared_data/bia_heyde/df_features_halophily_pipeline2.2_merged.csv')
df_salt = pd.read_csv('work/groups/VEO/shared_data/bia_heyde/df_salt_filtered-salinity_best_assembly_temp_pH_oxygen_merged_assemblies.csv')

#Define 'Best assembly' as index of dataframe
df_features['Best assembly'] = df_features.index
df_salt['Best assembly'] = df_salt.index

#Merge data with exact match
df_features = df_features.merge(df_salt, left_index=True, right_index=True, how='outer')

# Salvar o DataFrame em um novo arquivo CSV
df_features.to_csv('dados_bacterias_com_genomas.csv', index=False)

print("Dados salvos com sucesso!")

