import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Carregar o resultado do PCA
# Carregar os dados
df = pd.read_csv('/home/wi38kap/BacterialData/df_filtered.csv')
resultado_pca = pd.read_csv('/home/wi38kap/BacterialData/resultado_pca_filtered.csv')

y = df.iloc[:, -1]   # Última coluna
y= y.fillna(0)

# Pegar as duas primeiras colunas do resultado do PCA como X
X = resultado_pca.iloc[:, :2]
X = X.select_dtypes(include=[float])
X= X.fillna(0)


# Mapear as classes para valores numéricos
class_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_mapped = y.map(class_mapping)
print(y_mapped.head(2))


# Dividir os dados em conjunto de treino, teste e validação
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

# Instanciar e treinar o modelo XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Fazer previsões nos dados de teste e validação
y_pred_test = xgb_model.predict(X_test)
y_pred_val = xgb_model.predict(X_val)

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
metrics_df.to_csv('/home/wi38kap/BacterialData/metricas_avaliacao_Xgb.csv', index=False)

# Imprimir as métricas de avaliação
print(metrics_df)

# Plotar a matriz de confusão dos dados de teste
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão (Teste)\nAcurácia: {:.2f}".format(accuracy_test))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix_XGB_test.png')
plt.show()

# Plotar a matriz de confusão dos dados de validação
conf_matrix_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão (Validação)\nAcurácia: {:.2f}".format(accuracy_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix_XGB_val.png')
plt.show()
