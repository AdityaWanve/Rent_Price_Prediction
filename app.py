from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import gdown

# Model file info
MODEL_PATH = 'xgboost_model_analysis_pune.joblib'
MODEL_URL = 'https://drive.google.com/uc?id=10jJREi0P-DbrTRQ5u77-WuJLyBk4epNC'

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model and encoders
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load('label_encoders_pune.joblib')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedroom = int(request.form['bedroom'])
        layout_type = request.form['layout_type'].upper()
        property_type = request.form['property_type']
        locality = request.form['locality']
        furnish_type = request.form['furnish_type']
        bathroom = int(request.form['bathroom'])
        city = request.form['city']
        area = int(request.form['area'])
        seller_type = request.form['seller_type']

        # Normalize furnish_type
        if furnish_type.lower() in ['semi furnished', 'semi-furnished']:
            furnish_type = 'Semi-Furnished'
        elif furnish_type.lower() == 'unfurnished':
            furnish_type = 'Unfurnished'
        elif furnish_type.lower() == 'furnished':
            furnish_type = 'Furnished'

        # Normalize seller_type
        if seller_type.lower() == 'owner':
            seller_type = 'OWNER'
        elif seller_type.lower() == 'agent':
            seller_type = "AGENT"

        # Normalize city
        if city.lower() == 'pune':
            city = 'Pune'

        # Prepare input
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

        # Label encode categorical columns
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                if col in label_encoders:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                else:
                    return f"Label encoder for column {col} not found."

        # Predict
        prediction = model.predict(input_data)[0]
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"Error occurred: {str(e)}"

# Do not add app.run() here; Vercel handles that


# if __name__ == '__main__':
#     app.run(debug=True)
