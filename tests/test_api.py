from fastapi.testclient import TestClient

from moderaai.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_moderate_safe_text():
    resp = client.post("/moderate/text", json={"text": "Thanks for the update, looks great!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "safe"
    assert body["is_safe"] is True


def test_moderate_toxic_text():
    resp = client.post("/moderate/text", json={"text": "You are so stupid and worthless."})
    assert resp.status_code == 200
    assert resp.json()["label"] == "toxic"


def test_moderate_spam_text():
    resp = client.post("/moderate/text", json={"text": "FREE MONEY, click here now to claim your prize!!!"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "spam"


def test_moderate_rejects_empty_text():
    resp = client.post("/moderate/text", json={"text": ""})
    assert resp.status_code == 422
