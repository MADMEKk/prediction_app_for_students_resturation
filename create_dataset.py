import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import LabelEncoder

# -------------------- Part 1: Dataset Creation and Saving to SQLite -------------------- #

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

# Save the dataset into SQLite
conn = sqlite3.connect('meal_data.db')
df.to_sql('meal_data', conn, if_exists='replace', index=False)
conn.commit()

print("Dataset saved to SQLite database 'meal_data.db'")

# Save the encoders to SQLite
meal_type_encoder_data = {'class': meal_type_encoder.classes_.tolist()}
meal_encoder_data = {'class': meal_encoder.classes_.tolist()}

# Create a table for encoders if it doesn't exist
conn.execute('CREATE TABLE IF NOT EXISTS encoders (name TEXT, data TEXT)')

# Insert encoder data
conn.execute('INSERT INTO encoders (name, data) VALUES (?, ?)', ('meal_type_encoder', str(meal_type_encoder_data)))
conn.execute('INSERT INTO encoders (name, data) VALUES (?, ?)', ('meal_encoder', str(meal_encoder_data)))
conn.commit()

print("Encoders saved to SQLite database 'meal_data.db'")

conn.close()
