import json
import os
import secrets
import warnings
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = env_flag("SESSION_COOKIE_SECURE", False)

if not os.getenv("SECRET_KEY"):
    warnings.warn(
        "SECRET_KEY is not set. A temporary key was generated for this process. "
        "Set SECRET_KEY in production so sessions remain stable across restarts.",
        RuntimeWarning,
    )

db = SQLAlchemy(app)
csrf = CSRFProtect(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
SOIL_LABELS_FALLBACK = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=False, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

    def get_id(self):
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


def bootstrap_database():
    with app.app_context():
        db.create_all()

        if not ADMIN_EMAIL and not ADMIN_PASSWORD:
            return

        if not ADMIN_EMAIL or not ADMIN_PASSWORD:
            warnings.warn(
                "Both ADMIN_EMAIL and ADMIN_PASSWORD must be set to bootstrap an admin user.",
                RuntimeWarning,
            )
            return

        if len(ADMIN_PASSWORD) < 12:
            raise RuntimeError("ADMIN_PASSWORD must be at least 12 characters long.")

        admin_user = User.query.filter_by(email=ADMIN_EMAIL).first()
        if admin_user is None:
            admin_user = User(
                email=ADMIN_EMAIL,
                password_hash=generate_password_hash(ADMIN_PASSWORD),
                is_active=True,
                is_admin=True,
            )
            db.session.add(admin_user)
        else:
            admin_user.is_active = True
            admin_user.is_admin = True
            if not check_password_hash(admin_user.password_hash, ADMIN_PASSWORD):
                admin_user.password_hash = generate_password_hash(ADMIN_PASSWORD)

        db.session.commit()


def get_soil_labels():
    labels_path = BASE_DIR / "soil_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(labels, list) and labels:
            return labels

    train_dir = BASE_DIR / "dataset" / "Train"
    if train_dir.exists():
        labels = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
        if labels:
            return labels

    return SOIL_LABELS_FALLBACK


@lru_cache(maxsize=1)
def load_prediction_assets():
    import joblib
    import tensorflow as tf

    required_paths = {
        "soil model": BASE_DIR / "soil_cnn_model.h5",
        "crop model": BASE_DIR / "crop_model.h5",
        "soil encoder": BASE_DIR / "soil_encoder.pkl",
        "crop encoder": BASE_DIR / "crop_encoder.pkl",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction assets: {', '.join(missing)}")

    return {
        "soil_model": tf.keras.models.load_model(required_paths["soil model"], compile=False),
        "crop_model": tf.keras.models.load_model(required_paths["crop model"], compile=False),
        "soil_encoder": joblib.load(required_paths["soil encoder"]),
        "crop_encoder": joblib.load(required_paths["crop encoder"]),
        "soil_labels": get_soil_labels(),
    }


def predict_crop_from_upload(file_storage, temperature: float, rainfall: float):
    import numpy as np
    import pandas as pd
    from PIL import Image, UnidentifiedImageError

    if not -50 <= temperature <= 70:
        raise ValueError("Temperature must be between -50°C and 70°C.")
    if not 0 <= rainfall <= 10000:
        raise ValueError("Rainfall must be between 0 and 10000 mm.")

    assets = load_prediction_assets()

    try:
        image = Image.open(file_storage.stream)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    if image.width * image.height > 25_000_000:
        raise ValueError("The image is too large to process safely.")

    image = image.convert("RGB").resize((128, 128))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    soil_probabilities = assets["soil_model"].predict(image_batch, verbose=0)[0]
    soil_index = int(np.argmax(soil_probabilities))
    soil_labels = assets["soil_labels"]

    if soil_index >= len(soil_labels):
        raise RuntimeError("Soil model output does not match the configured soil labels.")

    soil_label = str(soil_labels[soil_index])

    encoder_categories = [str(value) for value in assets["soil_encoder"].categories_[0]]
    encoded_soil_label = next(
        (value for value in encoder_categories if value.casefold() == soil_label.casefold()),
        None,
    )
    if encoded_soil_label is None:
        raise RuntimeError(f"Soil label '{soil_label}' is not supported by the crop model.")

    soil_features = assets["soil_encoder"].transform(
        pd.DataFrame({"soil_type": [encoded_soil_label]})
    )
    weather_features = np.array([[temperature, rainfall]], dtype=np.float32)
    crop_features = np.concatenate([soil_features, weather_features], axis=1)

    crop_probabilities = assets["crop_model"].predict(crop_features, verbose=0)[0]
    crop_index = int(np.argmax(crop_probabilities))
    predicted_crop = str(assets["crop_encoder"].inverse_transform([crop_index])[0])

    return {
        "predicted_crop": predicted_crop,
        "soil_type": soil_label,
        "soil_confidence": round(float(soil_probabilities[soil_index]), 4),
        "crop_confidence": round(float(crop_probabilities[crop_index]), 4),
    }


bootstrap_database()


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Password must be at least 8 characters long.")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for("register"))

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            is_active=False,
            is_admin=False,
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! Wait for admin activation.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid credentials.")
            return redirect(url_for("login"))
        if not user.is_active:
            flash("Account not activated. Wait for admin approval.")
            return redirect(url_for("login"))

        login_user(user)
        return redirect(url_for("admin" if user.is_admin else "farmer"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/farmer")
@login_required
def farmer():
    if not current_user.is_active or current_user.is_admin:
        return redirect(url_for("login"))
    return render_template("farmer.html", user=current_user)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if not current_user.is_active or current_user.is_admin:
        return jsonify({"error": "Prediction access is available to active farmer accounts."}), 403

    soil_image = request.files.get("soil_image")
    if soil_image is None or not soil_image.filename:
        return jsonify({"error": "Please upload a soil image."}), 400
    if soil_image.mimetype and soil_image.mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        return jsonify({"error": "Supported image types are JPEG, PNG, and WebP."}), 400

    try:
        temperature = float(request.form.get("temperature", ""))
        rainfall = float(request.form.get("rainfall", ""))
        result = predict_crop_from_upload(soil_image, temperature, rainfall)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if not current_user.is_admin:
        return redirect(url_for("login"))

    if request.method == "POST":
        user_id = request.form.get("user_id", "")
        try:
            user = db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            user = None

        if user and not user.is_admin:
            user.is_active = True
            db.session.commit()
            flash(f"Activated {user.email}")
        return redirect(url_for("admin"))

    users = User.query.filter(User.is_admin.is_(False)).order_by(User.email).all()
    return render_template("admin.html", users=users)


@app.route("/farmers")
@login_required
def farmers():
    if not current_user.is_admin:
        return redirect(url_for("login"))

    activated_users = (
        User.query.filter(User.is_active.is_(True), User.is_admin.is_(False))
        .order_by(User.email)
        .all()
    )
    not_activated_users = (
        User.query.filter(User.is_active.is_(False), User.is_admin.is_(False))
        .order_by(User.email)
        .all()
    )
    return render_template(
        "farmers.html",
        activated_users=activated_users,
        not_activated_users=not_activated_users,
    )


@app.errorhandler(413)
def upload_too_large(_error):
    if request.path == "/predict":
        return jsonify({"error": "Uploaded image is too large."}), 413
    return "Request too large", 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        debug=env_flag("FLASK_DEBUG", False),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=port,
    )
