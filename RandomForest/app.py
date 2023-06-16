import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

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
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Step 5: Save the Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Step 5: Save the Vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
