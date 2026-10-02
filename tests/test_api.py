from fastapi.testclient import TestClient


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_spam(client):
    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You have won a free prize. Click here to claim your reward!"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "spam"
    assert 0 <= data["spam_probability"] <= 1


def test_predict_ham(client):
    response = client.post(
        "/predict",
        json={
            "message": "Hey, are we still meeting for lunch today?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "ham"
    assert 0 <= data["spam_probability"] <= 1


def test_predict_empty_message(client):
    response = client.post(
        "/predict",
        json={"message": ""},
    )

    assert response.status_code == 422