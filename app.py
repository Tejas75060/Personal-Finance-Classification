"""
Personal Finance Classification Web Application
Flask Application serving UI and REST API.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from src.predictor import FinancialClassifierService

app = Flask(__name__)

# Initialize prediction service
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

service = FinancialClassifierService(models_dir=MODELS_DIR)

# Load metrics and summary
with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
    METRICS_DATA = json.load(f)

with open(os.path.join(DATA_DIR, "eda_summary.json"), "r") as f:
    EDA_SUMMARY = json.load(f)

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=list(METRICS_DATA['results'].keys()),
        best_model=METRICS_DATA['best_model'],
        metrics=METRICS_DATA,
        eda=EDA_SUMMARY
    )

@app.route("/model-comparison")
def comparison():
    return render_template(
        "comparison.html",
        results=METRICS_DATA['results'],
        best_model=METRICS_DATA['best_model'],
        classes=METRICS_DATA['classes']
    )

@app.route("/eda")
def eda():
    return render_template(
        "eda.html",
        eda=EDA_SUMMARY
    )

@app.route("/report")
def report():
    return render_template(
        "report.html",
        results=METRICS_DATA['results'],
        best_model=METRICS_DATA['best_model'],
        eda=EDA_SUMMARY
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        model_name = data.get("model_name", METRICS_DATA['best_model'])
        prediction_result = service.predict(data, model_name=model_name)
        return jsonify({"success": True, "data": prediction_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(METRICS_DATA)

@app.route("/api/eda-summary", methods=["GET"])
def api_eda():
    return jsonify(EDA_SUMMARY)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
