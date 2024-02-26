# BacterialData
Bacterial datas

#1. Loading the Data
#The data is loaded from the CSV file dados_bacterias_com_genomas_group_Sall.csv located at /home/wi38kap/BacterialData/.
import pandas as pd
df = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv')

#2. Removing Columns
#Some unnecessary columns are removed from the DataFrame, including 'Unnamed: 0', 'Halophily', 'Class', and 'Species'.
df = df.drop(['Unnamed: 0', 'Halophily', 'Class', 'Species'], axis=1)

#3. Handling Missing Values
#The 'Salinity group' column is converted to the string data type, and any missing values are filled with 0.
df['Salinity group'] = df['Salinity group'].astype(str)
df['Salinity group'] = df['Salinity group'].fillna(0)

#4. Encoding Categorical Variables
#The 'Salinity group' column, containing classes in word format, is encoded into numerical values using scikit-learn's LabelEncoder.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Salinity group'] = le.fit_transform(df['Salinity group'])

#5. Data Splitting
#The data is split into features (X) and target variable (y).

X, y = df.loc[:, df.columns != 'Salinity group'], df['Salinity group']
