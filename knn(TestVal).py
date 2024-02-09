import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('df_salt_filtered-salinity_best_assembly - df_salt_filtered-salinity_best_assembly.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)


print('DataFrame')
print(df)

df['Best assembly'] = df['Best assembly'].astype(str)
df['Salt optimum'] = df['Salt optimum'].astype(str)
df['Salt all'] = df['Salt all'].astype(str)
df['Temperature'] = df['Temperature'].astype(str)
df['Low-bound pH'] = df['Low-bound pH'].astype(str)
df['Up-bound pH'] = df['Up-bound pH'].astype(str)
df['Oxygen tolerance'] = df['Oxygen tolerance'].astype(str)

# # Codificar colunas categóricas, se necessário
le_class = LabelEncoder()
le_species = LabelEncoder()
le_assembly = LabelEncoder()
le_salt_opt = LabelEncoder()
le_salt_all = LabelEncoder()
le_temperature = LabelEncoder()
le_low_bound = LabelEncoder()
le_up_bound = LabelEncoder()
le_oxygen = LabelEncoder()
le_taxa = LabelEncoder()
#
df['Class'] = le_class.fit_transform(df['Class'])
df['Species'] = le_species.fit_transform(df['Species'])
df['Best assembly'] = le_assembly.fit_transform(df['Best assembly'])
df['Salt optimum'] = le_salt_opt.fit_transform(df['Salt optimum'])
df['Salt all'] = le_salt_all.fit_transform(df['Salt all'])
df['Temperature'] = le_temperature.fit_transform(df['Temperature'])
df['Low-bound pH'] = le_low_bound.fit_transform(df['Low-bound pH'])
df['Up-bound pH'] = le_up_bound.fit_transform(df['Up-bound pH'])
df['Oxygen tolerance'] = le_oxygen.fit_transform(df['Oxygen tolerance'])
df['taxa'] = le_oxygen.fit_transform(df['taxa'])


# Separar as features (X) e a variável alvo (y)
X = df.drop('Temperature', axis=1)
y = df['Class']


print('Features')
print(X)
print('Variavel alvo')
print(y)


# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Dividir o conjunto de dados em treino (60%), teste (20%) e validação (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('X_train')
print(X_train)
print('X_test')
print(X_test)
print('y_train')
print(y_train)
print('y_test')
print(y_test)

# Verificar as formas dos conjuntos de dados
print("Treino:", X_train.shape, y_train.shape)
print("Teste:", X_test.shape, y_test.shape)
print("Validação:", X_val.shape, y_val.shape)



# Padronizar as features (importante para o KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val= scaler.transform(X_val)

print('X_train padronizado')
print(X_train)
print('X_test padronizado')
print(X_test)


# Criar o modelo KNN
knn= KNeighborsClassifier(n_neighbors=10)# Definir o número de vizinhos como 10


# Treinar o classificador usando os dados de treinamento
knn.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred_test = knn.predict(X_test)

# Calcular a acurácia nos dados de teste
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Acurácia nos dados de teste:", accuracy_test)

# Fazer previsões nos dados de validação
y_pred_val = knn.predict(X_val)

# Calcular a acurácia nos dados de validação
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Acurácia nos dados de validação:", accuracy_val)

# Calcular o recall para o conjunto de teste
recall_test = recall_score(y_test, y_pred_test, average='weighted')

# Calcular o F1-score para o conjunto de teste
f1_score_test = f1_score(y_test, y_pred_test, average='weighted')

# Imprimir o relatório de classificação
print('\nRelatório de Classificação:\n', classification_report(y_val, y_pred_val))


# Criar DataFrame a partir do relatório de classificação
report_dict = classification_report(y_val, y_pred_val, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

# Salvar o DataFrame em um arquivo CSV
#df_report.to_csv('relatorio_classificacaoKNN10TTV.csv', index=True)

# Calcular o recall para o conjunto de validação
recall_val = recall_score(y_val, y_pred_val, average='weighted')

# Calcular o F1-score para o conjunto de validação
f1_score_val = f1_score(y_val, y_pred_val, average='weighted')

print("Accuracy no conjunto de validação:", accuracy_val)
print("Recall no conjunto de validação:", recall_val)
print("F1-score no conjunto de validação:", f1_score_val)


# Definir as métricas (acurácia, recall e F1-score)
metrics = ['Accuracy', 'Recall', 'F1-score']
values_test = [accuracy_test, recall_test, f1_score_test]
values_val = [accuracy_val, recall_val, f1_score_val]

# Calcular o recall para o conjunto de teste
recall_test = recall_score(y_test, y_pred_test, average='weighted')

# Calcular o F1-score para o conjunto de teste
f1_score_test = f1_score(y_test, y_pred_test, average='weighted')

print("Recall no conjunto de teste:", recall_test)
print("F1-score no conjunto de teste:", f1_score_test)

# Criar um dataframe para facilitar o plot

df = pd.DataFrame({
    'Metric': metrics * 2,
    'Value': values_test + values_val,
    'Set': ['Teste'] * 3 + ['Validação'] * 3
})

# Plotar o gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', hue='Set', data=df, palette='viridis')
plt.title('Desempenho do Modelo nos Conjuntos de Teste e Validação')
plt.ylabel('Valor da Métrica')
plt.ylim(0, 1)  # Definir o limite do eixo y de 0 a 1 para representar a porcentagem
plt.legend(loc='upper right')

# Salvar o gráfico como um arquivo
plt.savefig('performance_plot.png')

# Exibir o gráfico
plt.show()
