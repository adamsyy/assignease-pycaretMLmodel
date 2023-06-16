import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        predict_data = request.json

        # Prepare the input data for prediction
        input_df = pd.DataFrame(predict_data)
        X_bidding_message_input = vectorizer.transform(input_df['bidding_message'])
        X_combined_input = pd.concat([input_df[['bid_amount']], pd.DataFrame(X_bidding_message_input.toarray())], axis=1, ignore_index=True)

        # Predict the combined score
        predictions = model.predict(X_combined_input)

        # Create a DataFrame with predictions and other information
        output_df = pd.DataFrame({'bid_amount': input_df['bid_amount'], 'bidding_message': input_df['bidding_message'], 'combined_score': predictions})

        # Sort the predictions in decreasing order of the score
        output_df = output_df.sort_values(by='combined_score', ascending=False)

        # Convert the DataFrame to a JSON response
        response = output_df.to_json(orient='records')

        return response

    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)
