
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Generating sample student data
np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "study_hours": np.random.uniform(1, 10, data_size),
    "attendance": np.random.uniform(40, 100, data_size),
    "previous_marks": np.random.uniform(30, 100, data_size),
    "sleep_hours": np.random.uniform(4, 10, data_size)
})

#Generating exam scores for our sample dataset
data["exam_score"] = (
    data["study_hours"] * 4 +
    data["attendance"] * 0.2 +
    data["previous_marks"] * 0.3 +
    data["sleep_hours"] * 1.5 +
    np.random.normal(0, 5, data_size)
).clip(0, 100)

#To Save the dataset
data.to_csv("students.csv", index=False)

#Separate input features and target
X = data.drop("exam_score", axis=1)
y = data["exam_score"]

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Training the ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

#Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Model training completed!")
print("Mean Absolute Error:", round(mae, 2))

#Prediction function
def predict_performance(study_hours, attendance,
                        previous_marks, sleep_hours):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "sleep_hours": sleep_hours
    }])

    prediction = model.predict(input_data)[0]

    return round(float(np.clip(prediction, 0, 100)), 2)
