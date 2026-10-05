import importlib
import os

os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.pop("ADMIN_EMAIL", None)
os.environ.pop("ADMIN_PASSWORD", None)

app_module = importlib.import_module("app")


def test_home_redirects_to_login():
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    response = client.get("/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")


def test_login_page_loads():
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    response = client.get("/login")

    assert response.status_code == 200
    assert b"Crop Recommendation System" in response.data


def test_prediction_requires_authentication():
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    response = client.post("/predict")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_security_headers_are_present():
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    response = client.get("/login")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
