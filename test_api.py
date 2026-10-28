
import json
from fastapi.testclient import TestClient
from app import app, feature_names

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["feature_count"] == 30


def test_prediction():
    with open("sample_input.json") as f:
        data = json.load(f)

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["B", "M"]
    assert result["diagnosis"] in ["Benign", "Malignant"]
    assert 0 <= result["confidence"] <= 1


def test_wrong_feature_count():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]}
    )

    assert response.status_code == 400
