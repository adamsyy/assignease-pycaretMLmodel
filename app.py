import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

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
X_combined.columns = X_combined.columns.astype(str)
df_combined = pd.concat([X_combined, y], axis=1)

# Load the pre-trained model
best_model = pickle.load(open("best_model.pkl", "rb"))

# Load the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X['bidding_message'])

# Define the Flask endpoint route and the function to handle the prediction
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request body
    input_data = request.json

    # Preprocess the input data
    input_df = pd.DataFrame(input_data)
    input_bidding_message = vectorizer.transform(input_df['bidding_message'])
    input_combined = pd.concat([input_df[['bid_amount']], pd.DataFrame(input_bidding_message.toarray())], axis=1)
    input_combined.columns = input_combined.columns.astype(str)

    # Make predictions using the loaded model
    predictions = best_model.predict(input_combined)
    
    # Combine predictions with input data
    output_data = []
    for i in range(len(input_data)):
        output_data.append({
            'bid_amount': input_data[i]['bid_amount'],
            'bidding_message': input_data[i]['bidding_message'],
            'combined_score': predictions[i]
        })

    # Sort output_data in decreasing order based on combined_score
    output_data = sorted(output_data, key=lambda x: x['combined_score'], reverse=True)

    # Return the sorted predictions as a JSON response
    return jsonify(output_data)

if __name__ == '__main__':
    app.run(debug=True)
