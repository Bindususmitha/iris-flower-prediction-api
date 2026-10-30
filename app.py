
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

saved_data = joblib.load("iris_model.joblib")
model = saved_data["model"]
target_names = saved_data["target_names"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris Flower Prediction API is Running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Provide flower measurements."}), 400

        features = data["features"]

        if not isinstance(features, list) or len(features) != 4:
            return jsonify({
                "error": "Exactly 4 measurements are required."
            }), 400

        values = [float(x) for x in features]

        if not all(np.isfinite(values)) or any(x <= 0 for x in values):
            return jsonify({"error": "Invalid measurements."}), 400

        result = int(model.predict([values])[0])

        return jsonify({
            "prediction": target_names[result],
            "class_number": result,
            "status": "Success"
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)))
