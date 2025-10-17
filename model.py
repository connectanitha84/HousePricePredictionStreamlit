# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Sample data (replace with real dataset)
data = pd.DataFrame({
    'area': [1000, 1500, 1200, 2000],
    'bedrooms': [2, 3, 2, 4],
    'bathrooms': [1, 2, 1, 3],
    'location': ['downtown', 'suburb', 'suburb', 'downtown'],
    'price': [50000, 80000, 60000, 100000]
})

X = data.drop('price', axis=1)
y = data['price']

# Preprocessing
numeric_features = ['area', 'bedrooms', 'bathrooms']
categorical_features = ['location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model_pipeline.fit(X, y)

# Save the whole pipeline
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Model & preprocessing pipeline saved as 'model.pkl'")
