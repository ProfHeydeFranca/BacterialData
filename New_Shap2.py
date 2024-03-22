import warnings
warnings.filterwarnings('ignore')
from sklearn import model_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carregar os dados
df = pd.read_csv('/home/wi38kap/BacterialData/df_filtered.csv')
#df = pd.read_csv(r'G:\Meu Drive\ProjetoAlemanha\Testes\df_filtered.csv', nrows=3, header=0)
# Remover colunas indesejadas
columns_to_drop = ['Best assembly_x', 'Salt opt min', 'Salt opt max', 'Salt all min', 'Salt all max']
df = df.drop(columns=columns_to_drop)

# Separar os dados de entrada (X) e saída (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Mapear as classes para valores numéricos
class_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_mapped = y.map(class_mapping)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_mapped, random_state=0)

# Treinar o modelo
cls = RandomForestClassifier(max_depth=2, random_state=0)
cls.fit(X_train, y_train)

# Plotar as importâncias das características
importances = cls.feature_importances_
indices = np.argsort(importances)
features = df.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('Relative_Importance.png')
plt.show()

# Calcular os valores SHAP
shap.initjs()
explainer = shap.TreeExplainer(cls)
shap_values = explainer.shap_values(X)

# Plotar gráficos de resumo SHAP
shap.summary_plot(shap_values, X.values, plot_type="bar", class_names=['low', 'medium', 'high'], feature_names=X.columns)
plt.savefig('shap_summary_plot_bar.png')
plt.show()

shap.summary_plot(shap_values[1], X.values, feature_names=X.columns)
plt.savefig('shap_summary_plot.png')
plt.show()

shap.dependence_plot(0, shap_values[0], X.values, feature_names=X.columns)
plt.savefig('shap_dependence_plot.png')
plt.show()

i = 8
shap.force_plot(explainer.expected_value[0], shap_values[0][i], X.values[i], feature_names=X.columns)
plt.savefig('shap_force_plot.png')
plt.show()

row = 8
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                     base_values=explainer.expected_value[0], 
                                     data=X_test.iloc[row],  
                                     feature_names=X_test.columns.tolist()))
plt.savefig('shap_waterfall_plot.png')
plt.show()

# Salvar os valores SHAP em um arquivo CSV
shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df.to_csv('shap_values.csv', index=False)

# Calcular a matriz de confusão
y_pred = cls.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['low', 'medium', 'high'], yticklabels=['low', 'medium', 'high'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_Shap.png')
plt.show()


# Selecionar as características mais importantes com base nos valores SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-10:]  # Escolha as 10 características mais importantes
X_train_selected = X_train.iloc[:, top_features_idx]
X_test_selected = X_test.iloc[:, top_features_idx]

# Treinar o modelo RandomForest com as características selecionadas
cls_selected = RandomForestClassifier(max_depth=2, random_state=0)
cls_selected.fit(X_train_selected, y_train)

# Realizar previsões no conjunto de teste
y_pred_selected = cls_selected.predict(X_test_selected)

# Calcular métricas de desempenho
accuracy = accuracy_score(y_test, y_pred_selected)
precision = precision_score(y_test, y_pred_selected, average='weighted')
recall = recall_score(y_test, y_pred_selected, average='weighted')
fscore = f1_score(y_test, y_pred_selected, average='weighted')

# Salvar métricas em CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F-Score'],
    'Score': [accuracy, precision, recall, fscore]
})
metrics_df.to_csv('performance_metrics.csv', index=False)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred_selected)

# Plotar a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['low', 'medium', 'high'], yticklabels=['low', 'medium', 'high'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Selected Features)')
plt.savefig('confusion_matrix_selected_features_Shap.png')
plt.show()