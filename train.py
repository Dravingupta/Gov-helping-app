import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Split features and label
X = data.drop("placed", axis=1)
y = data["placed"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

joblib.dump(model, "placement_model.pkl")
print("Model saved as placement_model.pkl 💾")


print("Model trained successfully ✅")

# Test prediction (new student)
sample_student = [[7.5, 6, 3, 1, 6]]
prediction = model.predict(sample_student)

print("Prediction (1 = Placed, 0 = Not Placed):", prediction[0])
