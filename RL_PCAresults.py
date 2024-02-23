from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
finalDf = pd.read_csv('/home/wi38kap/BacterialData/resultado_pca.csv',nrows=3)
finalDfy = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv',nrows=3)

# Selecionar as features e o target
Xfinal = finalDf[['Componente Principal 1', 'Componente Principal 2']]
yfinal = finalDfy['Salinity group'].fillna(0)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(Xfinal, yfinal, test_size=0.3, random_state=42)

# Instanciar e treinar o modelo de regressão logística
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# Fazer previsões
y_pred = logistic.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Salvar os resultados em um arquivo CSV
results = pd.DataFrame({"True": y_test, "Predicted": y_pred})
results.to_csv("logistic_regression_results.csv", index=False)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('/home/wi38kap/BacterialData/CF.png')
plt.show()

# Imprimir a acurácia
print("Acurácia:", accuracy)
