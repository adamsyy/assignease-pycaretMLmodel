import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from pycaret.regression import *

# Step 1: Load and Preprocess the Dataset
with open('dataset.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)
X = df[['bid_amount', 'bidding_message']]
y = df['combined_score']

# Step 2: Perform Feature Engineering
vectorizer = TfidfVectorizer()
X_bidding_message = vectorizer.fit_transform(X['bidding_message'])

# Step 3: Train a Machine Learning Model
X_combined = pd.concat([X[['bid_amount']], pd.DataFrame(X_bidding_message.toarray())], axis=1)
X_combined.columns = X_combined.columns.astype(str)  # Convert feature names to string
df_combined = pd.concat([X_combined, y], axis=1)

# Initialize PyCaret regression setup
regression_setup = setup(data=df_combined, target='combined_score')

# Compare models and select the best one
best_model = compare_models()

# Step 4: Fit the best model on the entire dataset
best_model.fit(X_combined, y)

# Save the best model using pickle
pickle.dump(best_model, open("best_model.pkl", "wb"))

# Step 5: Evaluate the Model
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Mean Squared Error:', mse)
print('R-squared:', r2)
