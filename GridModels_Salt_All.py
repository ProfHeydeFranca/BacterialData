import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import (H2ORandomForestEstimator, H2OGradientBoostingEstimator, 
                            H2ODeepLearningEstimator, H2OGeneralizedLinearEstimator, 
                            H2OSupportVectorMachineEstimator)

# Inicializar H2O
h2o.init()

# Carregar dados preparados
data = h2o.import_file("/home/wi38kap/BacterialData/pca_results.csv")

# Dividir os dados em treino e validação
train, valid, test = data.split_frame([0.6, 0.2], seed=42)

# Definir os modelos a serem avaliados
models = {
    "random_forest": H2ORandomForestEstimator,
    "gradient_boosting": H2OGradientBoostingEstimator,
    "deep_learning": H2ODeepLearningEstimator,
    "generalized_linear_model": H2OGeneralizedLinearEstimator,
    "svm_linear": H2OSupportVectorMachineEstimator,
    "svm_polynomial": H2OSupportVectorMachineEstimator,
    "svm_rbf": H2OSupportVectorMachineEstimator,
    "svm_sigmoid": H2OSupportVectorMachineEstimator
}

# Definir o grid de hiperparâmetros para cada modelo
params = {
    "random_forest": {"ntrees": [50, 100, 150], "max_depth": [5, 10, 15]},
    "gradient_boosting": {"ntrees": [50, 100, 150], "max_depth": [3, 5, 7], "learn_rate": [0.1, 0.01, 0.001]},
    "deep_learning": {"hidden": [[32, 32], [64, 64], [100, 100]]},
    "generalized_linear_model": {"alpha": [0.1, 0.5, 0.9], "lambda": [1e-5, 1e-6, 1e-7]},
    "svm_linear": {"kernel": "Linear"},
    "svm_polynomial": {"kernel": "Polynomial"},
    "svm_rbf": {"kernel": "Radial"},
    "svm_sigmoid": {"kernel": "Sigmoid"}
}

# Executar a grade de modelos
grid_searches = {}
for model_name, model_class in models.items():
    model = model_class()
    params_grid = params[model_name]
    grid = H2OGridSearch(model, hyper_params=params_grid)
    grid.train(x=data.col_names, y="Salt_all", training_frame=train, validation_frame=valid)
    grid_searches[model_name] = grid

# Analisar os resultados
for model_name, grid in grid_searches.items():
    print(f"{model_name} results:")
    print(grid.get_grid(sort_by="mse"))

    # Salvar resultados em CSV
    grid_df = grid.get_grid(sort_by="mse").as_data_frame()
    grid_df.to_csv(f"{model_name}_results.csv", index=False)

    # Plotar gráfico de visualização dos resultados
    sorted_models = grid_df.model_id.values
    mse_train_values = grid_df.mean_residual_deviance_train.values
    mse_valid_values = grid_df.mean_residual_deviance_valid.values
    plt.figure(figsize=(8, 6))
    plt.bar(sorted_models, mse_train_values, color='skyblue', label='Train')
    plt.bar(sorted_models, mse_valid_values, color='orange', label='Validation')
    plt.xlabel('Model')
    plt.ylabel('Mean Residual Deviance')
    plt.title(f'Mean Residual Deviance for {model_name}')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{model_name}_results_plot.png")  # Salvar o gráfico como imagem
    plt.show()

# Desligar H2O
h2o.shutdown()
