import pandas as pd

# Load your original dataset
df = pd.read_csv("data/student_mental_health.csv")

# Create mental_score
df["mental_score"] = df["depression"] + df["anxiety"] + df["isolation"]

# Function to assign burnout level
def assign_burnout(row):
    if row["mental_score"] >= 12 and row["academic_pressure"] >= 4:
        return 2  # Severe Burnout
    elif 7 <= row["mental_score"] <= 11:
        return 1  # Mild Burnout
    else:
        return 0  # No Burnout

# Apply the function
df["burnout_level"] = df.apply(assign_burnout, axis=1)

# Drop mental_score (optional)
df.drop("mental_score", axis=1, inplace=True)

# Save new dataset
df.to_csv("data/student_burnout_dataset.csv", index=False)

print("Burnout dataset created successfully!")
