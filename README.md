# prediction_app_for_students_resturation

# Meal Swipe Prediction System

## Description
This project is a complete pipeline for predicting the number of meal swipes in a dining facility based on the meal type, meal name, and day of the week. It includes dataset generation, storage, model training, and a Flask-based API for making predictions.

### Features:
1. **Data Generation and Storage**:
   - Synthetic data for meals and meal types from October to November 2023.
   - Data is stored in an SQLite database.
2. **Model Training**:
   - Random Forest Regressor is used to predict the number of swipes.
   - Evaluation metrics include Mean Squared Error (MSE) and R2 Score.
3. **Flask API**:
   - API for predicting swipes using the trained model.
   - `/test` route for health checks.
   - `/predict` route for making predictions.

---

## Installation

### Prerequisites
- Python 3.8+
- SQLite installed
- Pip for managing Python dependencies

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/meal-swipe-prediction.git
   cd meal-swipe-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

---

## Usage

### Dataset Creation
The script generates synthetic data for meals and meal types, saves the dataset into an SQLite database, and trains the model:
```bash
python data_creation_and_model_training.py
```

### Flask API
Start the Flask server:
```bash
python app.py
```

Make a POST request to the `/predict` endpoint with the following JSON payload:
```json
{
    "meal_type": 1,  // Encoded meal type
    "meal": 2,       // Encoded meal name
    "Day_of_Week": 3 // Day of the week (0 = Monday, 6 = Sunday)
}
```

### Example Request
Use `curl` or a tool like Postman:
```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"meal_type": 1, "meal": 2, "Day_of_Week": 3}'
```

---

## Project Structure
```
meal-swipe-prediction/
│
├── create_dataset.py                    # Dataset creation 
├── atrain.py                            #model training script
├── app.py                               # Flask├── atrain.py application
├── requirements.txt                     # Python dependencies
├── README.md                            # Documentation
├── meal_prediction_model.joblib         # Trained model
└── output_plot.png                      # Evaluation plot
```

---

## Future Improvements
- Extend the dataset to include more realistic variables like weather or holidays.
- Deploy the Flask API using a cloud service like AWS or Heroku.
- Add a front-end dashboard for data visualization.


