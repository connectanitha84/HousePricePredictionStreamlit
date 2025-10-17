from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved pipeline (includes preprocessing + model)
model = pickle.load(open('model.pkl', 'rb'))

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

        # 2️⃣ Convert to DataFrame (model expects same structure as training)
        input_df = pd.DataFrame([[area, bedrooms, bathrooms, location]],
                                columns=['area', 'bedrooms', 'bathrooms', 'location'])

        # 3️⃣ Prediction (preprocessing is applied automatically)
        prediction = model.predict(input_df)[0]

        # 4️⃣ Return result
        return render_template('result.html', prediction_text=f"💰 Estimated House Price: ₹{prediction:,.0f}")

if __name__ == "__main__":
    app.run()
