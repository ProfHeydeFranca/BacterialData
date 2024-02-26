# Import the H2OPrincipalComponentAnalysisEstimator from h2o.estimators
from h2o.estimators import H2OPrincipalComponentAnalysisEstimator

# Initialize H2O
h2o.init()

# Load Salinity and genomic features data into the dataframe
# Data_SalinityGen = h2o.upload_file("G:/Meu Drive/ProjetoAlemanha/Testes/dados_bacterias_com_genomas.csv")
Data_SalinityGen = h2o.upload_file("/home/wi38kap/BacterialData/BacterialData/dados_bacterias_com_genomas_merge.csv")
Data_SalinityGen.shape

# Convert column to a factor
Data_SalinityGen['Class'] = Data_SalinityGen['Class'].asfactor()
Data_SalinityGen['Class'].levels()
# Convert column to a factor
Data_SalinityGen['Species'] = Data_SalinityGen['Species'].asfactor()

# Apply ordinal encoding
Data_SalinityGen['Species'] = Data_SalinityGen['Species'].asnumeric()

# Convert pandas DataFrame to H2OFrame
Data_SalinityGen_h2o = h2o.H2OFrame(Data_SalinityGen)

# Select only the feature columns for PCA
features = Data_SalinityGen_h2o.columns
x = features.remove('Class')  # Remove the target column 'Class'
# Initialize and train the PCA model
pca = H2OPrincipalComponentAnalysisEstimator(k=10, transform="STANDARDIZE", pca_method="Power")
pca.train(x=x, training_frame=Data_SalinityGen_h2o)

# Get the loadings of the principal components
loadings = pca.varimp(use_pandas=True)

# Save the loadings to a CSV file
loadings.to_csv("loadings.csv", index=False)

# Check the importance of the principal components
print("Importance of principal components:")
print(loadings)

# Apply PCA to the data
pca_scores = pca.predict(Data_SalinityGen_h2o)

# Add the PCA scores to the original DataFrame
Data_SalinityGen_pca = Data_SalinityGen_h2o.cbind(pca_scores)

# Display the results
print("Original data with PCA scores:")
#print(Data_SalinityGen_pca.head())

# Save the PCA results to a CSV file
Data_SalinityGen_pca.as_data_frame().to_csv("pca_results.csv", index=False)

# Shutdown H2O
h2o.shutdown()
