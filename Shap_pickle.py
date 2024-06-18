import lightgbm as lgb
import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorar warnings
warnings.filterwarnings('ignore')

# Carregar os dados
file_path = r'/home/wi38kap/BacterialData/df_filtered.pickle'

# Ler o arquivo .pickle em um DataFrame
df = pd.read_pickle(file_path)

# Separar os dados de entrada (X) e saída (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Mapear as classes para valores numéricos
class_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_mapped = y.map(class_mapping)
print(y_mapped.head(2))

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mapped, test_size=0.2, random_state=7)

# Definir os parâmetros do modelo LightGBM
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True,
    "num_class": 3,
    "random_state": 7
}

# Treinar o modelo LightGBM
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
model = lgb.train(params, d_train, 100, valid_sets=[d_test])

# Explicador SHAP
explainer = shap.TreeExplainer(model)

# Calcular os valores SHAP para o conjunto de teste
shap_values = explainer.shap_values(X_test)

# Converter feature_names para uma lista
feature_names = X.columns.tolist()

# Plotar os gráficos de decisão SHAP para cada classe e salvar como .jpg
for class_index in range(len(shap_values)):
    expected_value = explainer.expected_value[class_index]  # Valor esperado para a classe atual
    shap_values_class = shap_values[class_index]  # Valores SHAP para a classe atual
    
    plt.figure()
    shap.decision_plot(expected_value, shap_values_class, X_test, link='logit', feature_names=feature_names)
    plt.title(f'SHAP Decision Plot for Class {class_index}')
    plt.savefig(f'/home/wi38kap/BacterialData/shap_decision_plot_class_{class_index}.jpg')
    plt.close()

# Calcular a importância das features usando SHAP e Gini
shap_importance = np.mean(np.abs(shap_values), axis=1)
gini_importance = model.feature_importance(importance_type='gain')

# Criar um DataFrame para armazenar as importâncias e as features
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean(|SHAP Value|)': shap_importance.mean(axis=0),
    'Gini Importance': gini_importance
})

# Scatterplot do VIP Score e o valor médio absoluto do SHAP com correlação de Pearson
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gini Importance', y='Mean(|SHAP Value|)', data=importance_df)
pearson_corr = importance_df[['Gini Importance', 'Mean(|SHAP Value|)']].corr().iloc[0, 1]
spearman_corr = importance_df[['Gini Importance', 'Mean(|SHAP Value|)']].corr(method='spearman').iloc[0, 1]
plt.title(f'Scatterplot of Gini Importance vs Mean(|SHAP Value|)\nPearson Corr: {pearson_corr:.2f}, Spearman Corr: {spearman_corr:.2f}')
plt.savefig('/home/wi38kap/BacterialData/scatter_gini_shap.jpg')
plt.close()

# Scatterplot do VIP Score e o valor médio absoluto do SHAP com correlação de Spearman
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mean(|SHAP Value|)', y='Mean(|SHAP Value|)', data=importance_df)
plt.title(f'Scatterplot of VIP Score vs Mean(|SHAP Value|)\nPearson Corr: {pearson_corr:.2f}, Spearman Corr: {spearman_corr:.2f}')
plt.savefig('/home/wi38kap/BacterialData/scatter_vip_shap.jpg')
plt.close()

# Salvar o DataFrame com as importâncias como .csv
output_csv_path = r'/home/wi38kap/BacterialData/Shap_Features.csv'
importance_df.to_csv(output_csv_path, index=False)

# Verificação para garantir que os arquivos foram salvos corretamente
print("Gráficos SHAP, scatterplots e DataFrame salvos com sucesso.")
