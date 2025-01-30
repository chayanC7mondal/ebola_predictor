from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get data from form
            total_cases = float(request.form["total_cases"])
            total_deaths = float(request.form["total_deaths"])
            days_since_start = float(request.form["days_since_start"])

            # Create input array
            input_data = np.array([[total_cases, total_deaths, days_since_start]])

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction = round(prediction, 4)  # Round for better display
        except ValueError:
            prediction = "Invalid input! Please enter numerical values."
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
