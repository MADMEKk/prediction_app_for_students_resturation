import joblib
import pandas as pd

# Load the saved model and encoders
model = joblib.load('meal_prediction_model.joblib')
meal_type_encoder = joblib.load('meal_type_encoder.joblib')
meal_encoder = joblib.load('meal_encoder.joblib')

# Input a manual test example
test_data = {
    'Date': '2023-10-15',  # Example date
    'Meal_Type': 'فطور',  # Breakfast
    'Meal': 'دجاج و ارز'  # Chicken and rice
}

# Preprocess the input data
day_of_week = pd.Timestamp(test_data['Date']).dayofweek  # Day of the week (0-6)
meal_type_encoded = meal_type_encoder.transform([test_data['Meal_Type']])[0]
meal_encoded = meal_encoder.transform([test_data['Meal']])[0]

# Prepare the input for the model
X_input = [[day_of_week, meal_type_encoded, meal_encoded]]

# Predict the swipes
predicted_swipes = model.predict(X_input)
print(f"Predicted Swipes: {int(predicted_swipes[0])}")
