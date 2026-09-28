from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_TRANSACTION = {
    "Time": 10000,
    "Amount": 50,
    "V1": 0,
    "V2": 0,
    "V3": 0,
    "V4": 0,
    "V5": 0,
    "V6": 0,
    "V7": 0,
    "V8": 0,
    "V9": 0,
    "V10": 0,
    "V11": 0,
    "V12": 0,
    "V13": 0,
    "V14": 0,
    "V15": 0,
    "V16": 0,
    "V17": 0,
    "V18": 0,
    "V19": 0,
    "V20": 0,
    "V21": 0,
    "V22": 0,
    "V23": 0,
    "V24": 0,
    "V25": 0,
    "V26": 0,
    "V27": 0,
    "V28": 0,
}


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_transaction(monkeypatch):
    def mock_predict_transaction(transaction):
        return {
            "fraud_probability": 0.0,
            "risk_score": 0.0,
            "risk_level": "Low Risk",
            "prediction": "NORMAL",
        }

    monkeypatch.setattr(
        "api.main.predict_transaction",
        mock_predict_transaction,
    )

    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert "prediction" in data


def test_predict_missing_fields():
    response = client.post(
        "/predict",
        json={
            "Time": 10000,
            "Amount": 50,
        },
    )

    assert response.status_code == 422


def test_predict_invalid_amount():
    invalid_transaction = VALID_TRANSACTION.copy()
    invalid_transaction["Amount"] = "hello"

    response = client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422
