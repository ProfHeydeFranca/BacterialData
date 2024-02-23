from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Carregar os dados
finalDf = pd.read_csv('/home/wi38kap/BacterialData/resultado_pca.csv',nrows=3)
finalDfy = pd.read_csv('/home/wi38kap/BacterialData/dados_bacterias_com_genomas_group_Sall.csv',nrows=3)

# Selecionar as features e o target
Xfinal = finalDf[['Componente Principal 1', 'Componente Principal 2']]
yfinal = finalDfy['Salinity group'].fillna(0)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(Xfinal, yfinal, test_size=0.3, random_state=42)

# Definir os modelos a serem avaliados
models = {
    "random_forest": RandomForestRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "neural_network": MLPRegressor(),
    "svm_linear": SVR(kernel='linear'),
    "svm_polynomial": SVR(kernel='poly'),
    "svm_rbf": SVR(kernel='rbf')
}

# Definir o grid de hiperparâmetros para cada modelo
params = {
    "random_forest": {"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15]},
    "gradient_boosting": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7], "learning_rate": [0.1, 0.01, 0.001]},
    "neural_network": {"hidden_layer_sizes": [(32,), (64,), (100,)]},
    "svm_linear": {},
    "svm_polynomial": {"degree": [2, 3, 4]},
    "svm_rbf": {"C": [0.1, 1, 10], "gamma": ['scale', 'auto']}
}

# Executar a busca em grade para cada modelo
grid_searches = {}
for model_name, model in models.items():
    params_grid = params[model_name]
    grid = GridSearchCV(model, params_grid, scoring='neg_mean_squared_error', cv=5)
    grid.fit(X_train, y_train)
    grid_searches[model_name] = grid

# Analisar os resultados
for model_name, grid in grid_searches.items():
    print(f"{model_name} results:")
    print(pd.DataFrame(grid.cv_results_))

    # Salvar resultados em CSV
    grid_df = pd.DataFrame(grid.cv_results_)
    grid_df.to_csv(f"{model_name}_results.csv", index=False)

    # Plotar gráfico de visualização dos resultados
plt.figure(figsize=(8, 6))
if model_name.startswith("svm"):
    param_name = 'param_C' if 'C' in params[model_name] else 'param_degree'
    x_values = grid_df[param_name].apply(str)  # Convertendo as tuplas em strings
    plt.plot(x_values, grid_df["mean_test_score"], marker='o')
    plt.xlabel('C' if 'C' in params[model_name] else 'Degree')
else:
    param_name = 'param_n_estimators' if 'n_estimators' in params[model_name] else 'param_hidden_layer_sizes'
    x_values = grid_df[param_name].apply(str)  # Convertendo as tuplas em strings
    plt.plot(x_values, grid_df["mean_test_score"], marker='o')
    plt.xlabel('Number of Estimators' if 'n_estimators' in params[model_name] else 'Hidden Layer Sizes')
plt.ylabel('Mean Negative MSE')
plt.title(f'Mean Negative MSE for {model_name}')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{model_name}_results_plot.png")  # Salvar o gráfico como imagem
plt.show()
