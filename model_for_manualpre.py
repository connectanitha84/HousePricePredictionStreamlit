# train_manual_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Sample data
data = pd.DataFrame({
    'area': [1000, 1500, 1200, 2000],
    'bedrooms': [2, 3, 2, 4],
    'bathrooms': [1, 2, 1, 3],
    'location': ['downtown', 'suburb', 'suburb', 'downtown'],
    'price': [50000, 80000, 60000, 100000]
})

X = data.drop('price', axis=1)
y = data['price']

# Numeric preprocessing
numeric_features = ['area', 'bedrooms', 'bathrooms']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Categorical preprocessing
categorical_features = ['location']
encoder = OneHotEncoder(sparse=False)
encoded_cat = encoder.fit_transform(X[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))

# Combine numeric + encoded categorical
X_preprocessed = pd.concat([X[numeric_features], encoded_cat_df], axis=1)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_preprocessed, y)

# Save objects
with open('regressor.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("✅ Model, scaler & encoder saved!")
