import pandas as pd
import numpy as np
import h2o
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from h2o.estimators import H2OPrincipalComponentAnalysis
import h2o


# Inicializar o H2O

h2o.init()

# Carregar os dados de fatores abióticos e features genomicas para o dataframe
Data_SalinityGen = h2o.upload_file("/work/groups/VEO/shared_data/bia_heyde/dados_bacterias_com_genomas_merge.csv")
Data_SalinityGen.shape

# Converter a coluna em um fator
Data_SalinityGen['Class'] = Data_SalinityGen['Class'].asfactor()
Data_SalinityGen['Class'].levels()
# Converter a coluna em um fator
Data_SalinityGen['Species'] = Data_SalinityGen['Species'].asfactor()

# Aplicar codificação ordinal
Data_SalinityGen['Species'] = Data_SalinityGen['Species'].asnumeric()

# Converter o DataFrame pandas para um H2OFrame
Data_SalinityGen_h2o = h2o.H2OFrame(Data_SalinityGen)

# Selecionar apenas as colunas de recursos para o PCA
features = Data_SalinityGen_h2o.columns
x = features.remove('Class')  # Remover a coluna de destino 'Class'

# Inicializar e treinar o modelo PCA
pca = H2OPrincipalComponentAnalysis(k=10, transform="STANDARDIZE", pca_method="Power")
pca.train(x=x, training_frame=Data_SalinityGen_h2o)

# Obter os loadings (cargas) dos componentes principais
loadings = pca.varimp(use_pandas=True)

# Verificar a importância dos componentes principais
print("Importância dos componentes principais:")
print(loadings)

# Aplicar o PCA aos dados
pca_scores = pca.predict(Data_SalinityGen_h2o)

# Adicionar os scores do PCA ao DataFrame original
Data_SalinityGen_pca = Data_SalinityGen_h2o.cbind(pca_scores)

# Exibir os resultados
print("Dados originais com scores do PCA:")
print(Data_SalinityGen_pca.head())

# Encerrar o H2O
h2o.shutdown()
