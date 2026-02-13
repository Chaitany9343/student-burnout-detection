import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Load dataset
df = pd.read_csv("data/student_burnout_dataset.csv")

#  Clean column names
df.columns = df.columns.str.strip()

#  Convert average_sleep to numeric
def convert_sleep(value):
    numbers = re.findall(r'\d+', str(value))
    if len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    elif len(numbers) == 1:
        return int(numbers[0])
    else:
        return 0

df["average_sleep"] = df["average_sleep"].apply(convert_sleep)

#  Select ONLY numeric features
features = [
    "average_sleep",
    "academic_workload",
    "academic_pressure",
    "financial_concerns",
    "social_relationships",
    "depression",
    "anxiety",
    "isolation",
    "future_insecurity"
]

X = df[features]
y = df["burnout_level"]

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#  Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#  Save model
joblib.dump(model, "models/burnout_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully!")
