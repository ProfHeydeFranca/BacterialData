from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')
#df = pd.read_csv(r"C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv", nrows=600)

df = df.drop(['Unnamed: 0','Halophily', 'Class','Species', 'Best assembly_y'], axis=1)
# Selecionar as features e o target
# Excluir os dados com salinidade igual a 0.0
# Contar quantas linhas possuem NaN na coluna 'Salinity group'
nan_count = df['Salinity group'].isna().sum()

# Imprimir o número de linhas com NaN na coluna 'Salinity group'
print("Número de linhas com NaN na coluna 'Salinity group' antes de deletar:", nan_count)
#data = df[df['Salinity group'] != 0.0]
df_filtered = df.dropna(subset=['Salinity group'])
#data=df.values
# Contar quantas linhas possuem NaN na coluna 'Salinity group'
nan_count = df_filtered ['Salinity group'].isna().sum()

# Imprimir o número de linhas com NaN na coluna 'Salinity group'
print("Número de linhas com NaN na coluna 'Salinity group' depois de deletar:", nan_count)
print(df_filtered .head())

# Salvar o DataFrame filtrado como um arquivo CSV
output_path = '/home/wi38kap/BacterialData/df_filtered.csv'
#df_filtered.to_csv(output_path, index=False)
#df_filtered.to_csv(r"C:\Users\00pau\df_filtered.csv", index=False)
df_filtered.to_csv(output_path, index=False)


