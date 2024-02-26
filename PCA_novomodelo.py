import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler        #Bağımlı,bağımsız değişkenler 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import LabelEncoder
from BorutaShap import BorutaShap, load_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=2)
df= pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')
df.head()
df = df.drop(['Unnamed: 0','Halophily', 'Class','Species'], axis=1)
df['Salinity group'] = df['Salinity group'].astype(str)
df['Salinity group'] = df['Salinity group'].fillna(0)

# If you want to replace NaN values with 0 in the entire DataFrame, you can use:
df.fillna(0, inplace=True)
#Split data into training features and labels

# Supondo que 'df' é o seu DataFrame e 'coluna' é o nome da coluna que contém as classes em forma de palavras
#le = LabelEncoder()
#df['Salinity group'] = le.fit_transform(df['Salinity group'])

target = ['Salinity group']
y = df[target]
print(df[target])
X = df.loc[:, df.columns  != 'Salinity group']
print(y)
print(target)
print(X)
#Scale Etmemiz gerekiyor;

x =StandardScaler().fit_transform(X)
print(x)#scale durumuna getirdik.

#PCA Projection 4 boyuttan 2 boyuta indirgeme:

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data= principalComponents , columns= ['Principal Component 1','Principal Component 2'])
print(principalDf) #veriyi istediğimiz şekile 2 boyutluya indirgedik.

#Target to PCA:

final_dataframe = pd.concat([principalDf,y],axis=1)
final_dataframe.head()
print(final_dataframe) #Verimizi son haline getirdik ve targeti ekledik.



#Görselleştirme:

targets = ['low','high', 'medium']
colors = ['g','b','r']

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df['Salinity group'] == target]
    plt.scatter(dftemp['Principal Component 1'], dftemp['Principal Component 2'], color = col)
plt.savefig('pca.png', dpi=300)  
plt.show()


#Varyans Koruma 

result = pca.explained_variance_ratio_
print(result)

total = pca.explained_variance_ratio_.sum()
print(total) # %96 veri seti korumak
