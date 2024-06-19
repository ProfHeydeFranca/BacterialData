rom sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import zstandard
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Caminho para o arquivo no Google Drive
file_path = r"/home/wi38kap/BacterialData/df_salt_kmer9_selected.pickle.zst"

# Lista para armazenar as linhas
lines = []

# Abrir o arquivo usando zstandard e pickle
with open(file_path, 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        # Deserializar os dados diretamente em uma lista de linhas
        lines = pickle.load(reader)

# Limitar a lista de linhas para as primeiras 200 linhas
#lines = lines[:200]

# Converter a lista de linhas em um DataFrame pandas
df = pd.DataFrame(lines)

# Exibir as primeiras linhas do DataFrame
print(df.head())

#df = df.drop(['Unnamed: 0','Halophily', 'Class','Species', 'Best assembly_y'], axis=1)
df = df.drop(['Salt all min', 'Salt opt max', 'Salt all max', 'Salt all mean', 'Salt opt min'], axis=1)

# Remover as linhas onde 'Salinity group' é igual a 'nan'
df = df[df['Salinity group'] != 'nan']





# Salvar o DataFrame filtrado como um arquivo CSV
#output_path = '/home/wi38kap/BacterialData/df_filtered.csv'
output_path = r"/home/wi38kap/BacterialData/df_filtered_total.csv"
#df_filtered.to_csv(output_path, index=False)
#df_filtered.to_csv(r"C:\Users\00pau\df_filtered.csv", index=False)
df.to_csv(output_path, index=False)

#salvar como pickle 
output_path = r'/home/wi38kap/BacterialData/df_filtered.pickle'
with open(output_path, 'wb') as f:
    pickle.dump(df, f)

