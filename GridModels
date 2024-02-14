import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2ODeepLearningEstimator, H2OGeneralizedLinearEstimator

# Initialize H2O
h2o.init()

# Load prepared data
data = h2o.import_file("/home/wi38kap/BacterialData/pca_results.csv")

# Define the models to be evaluated
models = {
    "random_forest": H2ORandomForestEstimator,
    "gradient_boosting": H2OGradientBoostingEstimator,
    "deep_learning": H2ODeepLearningEstimator,
    "generalized_linear_model": H2OGeneralizedLinearEstimator
}

# Define the hyperparameter grid for each model
params = {
    "random_forest": {"ntrees": [50, 100, 150], "max_depth": [5, 10, 15]},
    "gradient_boosting": {"ntrees": [50, 100, 150], "max_depth": [3, 5, 7], "learn_rate": [0.1, 0.01, 0.001]},
    "deep_learning": {"hidden": [[32, 32], [64, 64], [100, 100]]},
    "generalized_linear_model": {"alpha": [0.1, 0.5, 0.9], "lambda": [1e-5, 1e-6, 1e-7]}
}

# Run the grid of models
grid_searches = {}
for model_name, model_class in models.items():
    model = model_class()
    params_grid = params[model_name]
    grid = H2OGridSearch(model, hyper_params=params_grid)
    grid.train(x=data.col_names, y="target_column", training_frame=data)
    grid_searches[model_name] = grid

# Analyze the results
for model_name, grid in grid_searches.items():
    print(f"{model_name} results:")
    print(grid.get_grid(sort_by="mse"))

# Shutdown H2O
h2o.shutdown()
