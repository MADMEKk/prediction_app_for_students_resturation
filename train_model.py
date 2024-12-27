import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Define meals and meal types
meals = ['دجاج و ارز', 'عدس و بيض', 'سمك و ارز', 'معكرونة', 'شوربة']
meal_types = ['فطور', 'عشاء']  # Breakfast and Dinner

# Generate date range for October and November 2023
dates = pd.date_range(start='2023-10-01', end='2023-11-30', freq='D')
data = []

# Generate synthetic data
for date in dates:
    for meal_type in meal_types:
        meal = np.random.choice(meals)
        if meal_type == 'فطور':  # Breakfast
            swipes = np.random.poisson(lam=600)
        else:  # Dinner
            swipes = np.random.poisson(lam=450)
        data.append({'Date': date, 'Meal_Type': meal_type, 'Meal': meal, 'Swipes': swipes})

# Create DataFrame
df = pd.DataFrame(data)

# Add new features: Day of the week
df['Day_of_Week'] = df['Date'].dt.dayofweek  # This is numeric (0-6)

# Encode categorical features
meal_type_encoder = LabelEncoder()
meal_encoder = LabelEncoder()

# Fit encoders with all possible values
meal_type_encoder.fit(meal_types)
meal_encoder.fit(meals)

# Transform the data
df['Meal_Type_Encoded'] = meal_type_encoder.transform(df['Meal_Type'])
df['Meal_Encoded'] = meal_encoder.transform(df['Meal'])

# Define features (X) and target (y)
X = df[['Day_of_Week', 'Meal_Type_Encoded', 'Meal_Encoded']]  # Use Day_of_Week as is
y = df['Swipes']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

import matplotlib.pyplot as plt
# Predict values for the test set
y_pred = model.predict(X_test)

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Swipes')
plt.ylabel('Predicted Swipes')
plt.title('Actual vs Predicted Swipes')
plt.savefig('output_plot.png')  # Save the figure to a file
print("Plot saved as 'output_plot.png'")


# Save the model and encoders
joblib.dump(model, 'meal_prediction_model.joblib')
joblib.dump(meal_type_encoder, 'meal_type_encoder.joblib')
joblib.dump(meal_encoder, 'meal_encoder.joblib')
