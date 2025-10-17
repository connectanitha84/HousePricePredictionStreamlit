# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# ----------------------------
# Sample dataset
# ----------------------------
data = pd.DataFrame({
    'area': [1000, 1500, 1200, 2000, 2500, 1800, 1600, 2200],
    'bedrooms': [2, 3, 2, 4, 4, 3, 3, 4],
    'bathrooms': [1, 2, 1, 3, 3, 2, 2, 3],
    'location': ['downtown', 'suburb', 'suburb', 'downtown', 'suburb', 'downtown', 'suburb', 'downtown'],
    'price': [50000, 80000, 60000, 100000, 120000, 90000, 85000, 110000]
})

# ----------------------------
# Split features & target
# ----------------------------
X = data.drop('price', axis=1)
y = data['price']

# ----------------------------
# Preprocessing
# ----------------------------
numeric_features = ['area', 'bedrooms', 'bathrooms']
categorical_features = ['location']

# 1️⃣ Numeric scaler
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 2️⃣ Categorical encoding using get_dummies
# ----------------------------
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False,dtype=int)  # keep all columns

# Now X_encoded contains both numeric and one-hot encoded categorical features
X_preprocessed = X_encoded.copy()
print(X_preprocessed)

# # Combine numeric + encoded categorical
# X_preprocessed = pd.concat([X[numeric_features], encoded_cat_df], axis=1)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_preprocessed, y)

# ----------------------------
# Save model & preprocessors
# ----------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# with open('encoder.pkl', 'wb') as f:
#     pickle.dump(encoder, f)

print("✅ Model, scaler, and encoder saved as .pkl files!")
