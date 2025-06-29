from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import requests

MODEL_PATH = 'xgboost_model_analysis_pune.joblib'
MODEL_URL = 'https://drive.google.com/file/d/10jJREi0P-DbrTRQ5u77-WuJLyBk4epNC/view?usp=sharing'

# Auto-download if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)

# Load the model
model = joblib.load(MODEL_PATH)


# Load the model and label encoders
label_encoders = joblib.load('label_encoders_pune.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Load index.html file

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the request
    bedroom = int(request.form['bedroom'])
    layout_type = request.form['layout_type']
    property_type = request.form['property_type']
    locality = request.form['locality']
    furnish_type = request.form['furnish_type']
    bathroom = int(request.form['bathroom'])
    city = request.form['city']
    area = int(request.form['area'])
    seller_type = request.form['seller_type']

    # Standardize layout_type to uppercase (BHK/RK)
    layout_type = layout_type.upper()

    # Standardize furnish_type (Unfurnished, Semi-Furnished, Furnished) with hyphens
    if furnish_type.lower() == 'unfurnished':
        furnish_type = 'Unfurnished'
    elif furnish_type.lower() in ['semi furnished', 'semi-furnished']:
        furnish_type = 'Semi-Furnished'
    elif furnish_type.lower() == 'furnished':
        furnish_type = 'Furnished'

    # Ensure seller_type (owner) is in uppercase
    if seller_type.lower() == 'owner':
        seller_type = 'OWNER'
    elif seller_type.lower() == 'agent':
        seller_type = "AGENT"

    # Ensure city is in uppercase
    if city.lower() == 'pune':
        city = 'Pune'

    # Prepare input data for the model
    input_data = pd.DataFrame({
        'seller_type': [seller_type],
        'bedroom': [bedroom],
        'layout_type': [layout_type],
        'property_type': [property_type],
        'locality': [locality],
        'area': [area],
        'furnish_type': [furnish_type],
        'bathroom': [bathroom],
        'city': [city]
    })

    # Apply label encoders for categorical columns
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
            else:
                return f"Label encoder for column {col} not found."

    # Make prediction using the model
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
