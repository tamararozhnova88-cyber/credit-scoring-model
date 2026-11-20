from fastapi.testclient import TestClient
from src.api.app import app


def test_health_endpoint():
    response = TestClient(app).get("/")
    assert response.status_code == 200


def test_bad_request():
    response = TestClient(app).post("/predict", json={"AGE": 12})
    assert response.status_code == 422
