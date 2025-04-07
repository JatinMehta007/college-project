import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data/raw/insurance.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical variables
le = preprocessing.LabelEncoder()
data["Sex"] = le.fit_transform(data["sex"])
data["Smoker"] = le.fit_transform(data["smoker"])
data["Region"] = le.fit_transform(data["region"])

# Select features and target variable
X = data[["age", "bmi", "children", "Sex", "Smoker", "Region"]]
y = data["charges"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train_scaled, y_train)

# Evaluate the model
score = model.score(X_test_scaled, y_test)
print(f"Model Accuracy (R² Score): {score:.4f}")

# Save the trained model
model_path = "data/model/trained_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Trained model saved at: {model_path}")

# Load and test the saved model
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

# Sample prediction
sample_input = [[18, 18, 0, 0, 0, 2]]  # Example input
sample_input_scaled = scaler.transform(sample_input)
prediction = max(0, loaded_model.predict(sample_input_scaled)[0])
print(f"Predicted Insurance Cost: {prediction:.2f}")