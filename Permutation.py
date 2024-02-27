from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from eli5.sklearn import PermutationImportance

df = pd.read_csv(r'C:\Users\00pau\dados_bacterias_com_genomas_group_Sall.csv', nrows=3, header=0)
df = df.drop(['Unnamed: 0','Halophily', 'Class','Species'], axis=1)
df['Salinity group'] = df['Salinity group'].astype(str)
df['Salinity group'] = df['Salinity group'].fillna(0)
df.fillna(0, inplace=True)

le = LabelEncoder()
df['Salinity group'] = le.fit_transform(df['Salinity group'])

target = 'Salinity group'
y = df[target]
X = df.drop(target, axis=1)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting model
gbm = GradientBoostingRegressor()
gbm.fit(X_train, y_train)

# Use Permutation Importance to get feature importances
perm = PermutationImportance(gbm, scoring='neg_mean_squared_error', n_iter=2, random_state=42)
perm.fit(X_train, y_train)

# Get feature importances and prediction errors
importances = perm.feature_importances_
mse = mean_squared_error(y_test, gbm.predict(X_test))

# Plot the Variable Importance Plot (VIP)
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Variable Importance in Projection')
plt.savefig('VarImp1.png', dpi=300)
plt.show()

# Define the threshold
threshold = 0.05  # Example threshold value

# Select only the most important features
X_train_selected = X_train[:, importances > threshold]
X_test_selected = X_test[:, importances > threshold]

# Train a linear regression model with the selected features
regression_model = LinearRegression()
regression_model.fit(X_train_selected, y_train)

# Evaluate the model
score = regression_model.score(X_test_selected, y_test)
print("Model Score:", score)
