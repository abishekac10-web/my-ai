from flask import Flask, request, jsonify
import joblib

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
model = joblib.load("logistic_regression_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validation
        if not data or 'text' not in data or not data['text'].strip():
            return jsonify({'error': 'No text provided or invalid input format.'}), 400

        user_text = data['text']

        # Vectorize input text
        text_vectorized = vectorizer.transform([user_text])

        # Predict intent
        predicted_intent = model.predict(text_vectorized)[0]

        return jsonify({'intent': predicted_intent})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    print("Flask API starting…")
    app.run(host="0.0.0.0", port=5000)

