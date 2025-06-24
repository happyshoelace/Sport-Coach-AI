from flask import Flask, request, jsonify
import requests


app = Flask(__name__)


WSL_APP_URL = "http://127.0.0.1:6000"  # The URL of your WSL Flask app


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "prompt" not in data or "hist_data" not in data:
        return jsonify({"error": "Missing required fields in request"}), 400

    # Forward the request to the WSL Flask app
    try:
        response = requests.post(f"{WSL_APP_URL}/chat", json=data)
        return jsonify(response.json())
    except requests.ConnectionError:
        return jsonify({"error": "Unable to connect to WSL Flask app"}), 500


@app.route("/health", methods=["GET"])
def health():
    # Forward the health check to the WSL Flask app
    try:
        response = requests.get(f"{WSL_APP_URL}/health")
        return jsonify(response.json())
    except requests.ConnectionError:
        return jsonify({"error": "Unable to connect to WSL Flask app"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)