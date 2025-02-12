import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = sns.load_dataset("iris")

# Basic statistics
print(df.head())
print(df.describe())

# Visualizing the dataset
sns.pairplot(df, hue="species", diag_kind="kde")
plt.show()

# Boxplot for better understanding
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, orient="h")
plt.title("Iris Feature Distributions")
plt.show()

# Encode target variable
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])  # Convert species names to numeric labels

# Split dataset
X = df.drop(columns=["species"])
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline with preprocessing and model
pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoder for later use in API
with open("iris_model.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(le, encoder_file)

print("Model and encoder saved successfully!")
