# app.py (for UNSW-NB15)
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # To handle CORS requests

# ✅ Load the saved model and preprocessing tools
model = joblib.load("rf_model1_unsw.pkl")
scaler = joblib.load("scaler1_unsw.pkl")
pca = joblib.load("pca1_unsw.pkl")

# Optional: Load the label encoder to map prediction to string label (if you saved it)
# label_encoder = joblib.load("label_encoder_unsw.pkl")  # Uncomment if available

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin API access

@app.route("/", methods=["GET"])
def index():
    return "🚀 IDS API for UNSW-NB15 is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Receive input JSON containing the 'features' key
        input_data = request.json.get('features', [])

        # 2. Validate the number of features
        expected_features = scaler.mean_.shape[0]
        if len(input_data) != expected_features:
            return jsonify({
                "error": f"❌ Expected {expected_features} features, but received {len(input_data)}"
            })

        # 3. Preprocess: scale and apply PCA
        sample = np.array(input_data).reshape(1, -1)
        scaled = scaler.transform(sample)
        reduced = pca.transform(scaled)

        # 4. Predict
        prediction = model.predict(reduced)[0]

        # Optional: Convert numeric label to original string (if label_encoder is available)
        # prediction_label = label_encoder.inverse_transform([prediction])[0]

        # 5. Return response
        return jsonify({
            "prediction": int(prediction),
            # "label": prediction_label  # Uncomment if using label encoder
        })

    except Exception as e:
        return jsonify({"error": f"⚠️ {str(e)}"})

if __name__ == "__main__":
    print("✅ Starting Flask API for UNSW-NB15...")
    app.run(debug=True, host="0.0.0.0", port=5000)
