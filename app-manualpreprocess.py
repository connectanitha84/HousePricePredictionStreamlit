from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved objects
model = pickle.load(open('regressor.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1️⃣ Get form values
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        location = request.form['location']

        # 2️⃣ Create DataFrame
        input_df = pd.DataFrame([[area, bedrooms, bathrooms, location]],
                                columns=['area', 'bedrooms', 'bathrooms', 'location'])

        # 3️⃣ Preprocessing

        # Numeric
        numeric_features = ['area', 'bedrooms', 'bathrooms']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Categorical
        encoded_cat = encoder.transform(input_df[['location']])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['location']))

        # Combine
        input_preprocessed = pd.concat([input_df[numeric_features], encoded_cat_df], axis=1)

        # 4️⃣ Prediction
        prediction = model.predict(input_preprocessed)[0]

        return render_template('result.html', prediction_text=f"💰 Estimated House Price: ₹{prediction:,.0f}")

if __name__ == "__main__":
    app.run(debug=True)
