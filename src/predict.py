import joblib
import numpy as np
import re

# 1️⃣ Load trained model and scaler
model = joblib.load("models/burnout_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 2️⃣ Function to convert sleep text to number
def convert_sleep(value):
    numbers = re.findall(r'\d+', str(value))
    if len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    elif len(numbers) == 1:
        return int(numbers[0])
    else:
        return 0

# 3️⃣ Take input from user
print("Enter Student Details:\n")

sleep = input("Average Sleep (e.g., 4-6 hrs): ")
academic_workload = int(input("Academic Workload (1-5): "))
academic_pressure = int(input("Academic Pressure (1-5): "))
financial_concerns = int(input("Financial Concerns (1-5): "))
social_relationships = int(input("Social Relationships (1-5): "))
depression = int(input("Depression (1-5): "))
anxiety = int(input("Anxiety (1-5): "))
isolation = int(input("Isolation (1-5): "))
future_insecurity = int(input("Future Insecurity (1-5): "))

# 4️⃣ Convert sleep
sleep_numeric = convert_sleep(sleep)

# 5️⃣ Prepare input array
new_student = np.array([[
    sleep_numeric,
    academic_workload,
    academic_pressure,
    financial_concerns,
    social_relationships,
    depression,
    anxiety,
    isolation,
    future_insecurity
]])

# 6️⃣ Scale input
new_student_scaled = scaler.transform(new_student)

# 7️⃣ Predict
prediction = model.predict(new_student_scaled)[0]

# 8️⃣ Interpret result
if prediction == 0:
    print("\nPrediction: No Burnout ✅")
elif prediction == 1:
    print("\nPrediction: Mild Burnout ⚠")
else:
    print("\nPrediction: Severe Burnout 🚨")
