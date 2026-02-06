import joblib
import pandas as pd

model = joblib.load("placement_model.pkl")

student = pd.DataFrame([{
    "cgpa": 8.1,
    "dsa": 7,
    "projects": 4,
    "internship": 1,
    "communication": 7
}])

prediction = model.predict(student)
print("Prediction:", prediction[0])
