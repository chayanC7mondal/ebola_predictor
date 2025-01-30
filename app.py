# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained XGBoost model
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        total_cases = int(request.form['total_cases'])
        total_deaths = int(request.form['total_deaths'])
        days_since_start = int(request.form['days_since_start'])

        # Predict the CFR
        input_data = np.array([[total_cases, total_deaths, days_since_start]])
        predicted_cfr = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f"Predicted CFR: {predicted_cfr:.4f}")

if __name__ == "__main__":
    app.run(debug=True)
