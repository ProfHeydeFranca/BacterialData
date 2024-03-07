from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv(r"C:\Users\00pau\df_filtered.csv", nrows=600)

X = df.iloc[:, :-1]  # Todas as colunas exceto a última
y = df.iloc[:, -1]   # Última coluna

X = X.select_dtypes(include=[float])
X = X.fillna(0)
y = y.fillna(0)
print("head X:", X.head())
print("head y:", y.head())

# Mapear as classes para valores numéricos
class_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_mapped = y.map(class_mapping)
print(y_mapped.head(2))

print("qtd y depois de filtrado", y_mapped.value_counts())
#print(y_mapped)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Dividir o conjunto de dados em treino (60%), teste (20%) e validação (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y_mapped, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Shape de X_train:", X_train.shape)
print("Shape de X_test:", X_test.shape)
print("Shape de X_val:", X_val.shape)
print("Shape de y_train:", y_train.shape)
print("Shape de y_test:", y_test.shape)
print("Shape de y_val:", y_val.shape)

# Instanciar e treinar o modelo XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred_test = xgb_model.predict(X_test)

# Fazer previsões nos dados de validação
y_pred_val = xgb_model.predict(X_val)

# Calcular a acurácia do modelo XGBoost nos dados de teste
accuracy_test = accuracy_score(y_test, y_pred_test)

# Calcular a acurácia do modelo XGBoost nos dados de validação
accuracy_val = accuracy_score(y_val, y_pred_val)

# Calcular a matriz de confusão nos dados de teste
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# Calcular a matriz de confusão nos dados de validação
conf_matrix_val = confusion_matrix(y_val, y_pred_val)

# Calcular as métricas de avaliação nos dados de teste
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')

# Calcular as métricas de avaliação nos dados de validação
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val, average='weighted')
recall_val = recall_score(y_val, y_pred_val, average='weighted')
f1_val = f1_score(y_val, y_pred_val, average='weighted')

# Criar um DataFrame com as métricas de avaliação
metrics_df = pd.DataFrame({
    'Dataset': ['Teste', 'Validação'],
    'Acurácia': [accuracy_test, accuracy_val],
    'Precisão': [precision_test, precision_val],
    'Recall': [recall_test, recall_val],
    'F-Score': [f1_test, f1_val]
})

# Salvar as métricas de avaliação em uma tabela CSV
metrics_df.to_csv('metricas_avaliacao_xgb.csv', index=False)
# Plotar a matriz de confusão dos dados de teste
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusão (Teste)\nAcurácia: {accuracy_test:.2f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix_XGBoost_test.png')
plt.show()

# Plotar a matriz de confusão dos dados de validação
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusão (Validação)\nAcurácia: {accuracy_val:.2f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix_XGBoost_validation.png')
plt.show()

# Imprimir a acurácia nos dados de teste e validação
print("Acurácia (Teste):", accuracy_test)
print("Acurácia (Validação):", accuracy_val)
