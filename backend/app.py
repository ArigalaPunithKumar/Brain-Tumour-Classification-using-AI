import base64
import os
import secrets
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("COOKIE_SECURE", "true").lower() == "true"

database_url = os.getenv("DATABASE_URL", "").strip()
if not database_url:
    raise RuntimeError("DATABASE_URL is required. Add your Aiven MySQL connection string to Render.")

if database_url.startswith("mysql://"):
    database_url = database_url.replace("mysql://", "mysql+pymysql://", 1)
elif database_url.startswith("mysql+mysqlconnector://"):
    database_url = database_url.replace("mysql+mysqlconnector://", "mysql+pymysql://", 1)

# Aiven's Service URI can include ssl-mode=REQUIRED. PyMySQL expects SSL
# settings through connect_args, so remove that URI-only option first.
parts = urlsplit(database_url)
query = dict(parse_qsl(parts.query, keep_blank_values=True))
ssl_mode = query.pop("ssl-mode", query.pop("ssl_mode", "")).upper()
database_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
db_connect_args = {"ssl": {}} if ssl_mode in {"REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY"} else {}

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
    "connect_args": db_connect_args,
}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
allowed_origins = [frontend_url]
extra_origins = os.getenv("EXTRA_CORS_ORIGINS", "")
allowed_origins.extend([x.strip().rstrip("/") for x in extra_origins.split(",") if x.strip()])

CORS(
    app,
    supports_credentials=True,
    origins=allowed_origins,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    status = db.Column(db.String(20), nullable=False, default="active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = tv_models.mobilenet_v2(weights=None)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


def download_if_missing(path: Path, url: str):
    if path.exists() and path.stat().st_size > 0:
        return
    if not url:
        raise RuntimeError(f"Missing model {path.name} and no download URL was configured.")
    print(f"Downloading {path.name}...")
    with urlopen(url, timeout=300) as response, open(path, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"Downloaded {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


MODEL_URLS = {
    "classification": os.getenv(
        "CLASSIFICATION_MODEL_URL",
        "https://raw.githubusercontent.com/ArigalaPunithKumar/Brain-Tumour-Classification-using-AI/main/Frontend/life-care/mobilenet.pt",
    ),
    "relevance": os.getenv(
        "RELEVANCE_MODEL_URL",
        "https://raw.githubusercontent.com/ArigalaPunithKumar/Brain-Tumour-Classification-using-AI/main/Frontend/life-care/mobilenet_irrelevent.pt",
    ),
    "segmentation": os.getenv(
        "SEGMENTATION_MODEL_URL",
        "https://raw.githubusercontent.com/ArigalaPunithKumar/Brain-Tumour-Classification-using-AI/main/Frontend/life-care/best_model.pth",
    ),
}

classification_path = MODEL_DIR / "mobilenet.pt"
relevance_path = MODEL_DIR / "mobilenet_irrelevent.pt"
segmentation_path = MODEL_DIR / "best_model.pth"


loaded_model_name = None
loaded_model = None


def unload_model():
    global loaded_model_name, loaded_model
    loaded_model = None
    loaded_model_name = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(name):
    global loaded_model_name, loaded_model

    if loaded_model_name == name and loaded_model is not None:
        return loaded_model

    unload_model()

    if name == "relevance":
        download_if_missing(relevance_path, MODEL_URLS["relevance"])
        model = MobileNetModel(2)
        model.load_state_dict(torch.load(relevance_path, map_location=device))
    elif name == "classification":
        download_if_missing(classification_path, MODEL_URLS["classification"])
        model = MobileNetModel(2)
        model.load_state_dict(torch.load(classification_path, map_location=device))
    elif name == "segmentation":
        download_if_missing(segmentation_path, MODEL_URLS["segmentation"])
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation=None,
        )
        model.load_state_dict(torch.load(segmentation_path, map_location=device))
    else:
        raise ValueError(f"Unknown model: {name}")

    loaded_model = model.to(device).eval()
    loaded_model_name = name
    return loaded_model


def user_payload(user):
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "mobile": user.mobile,
        "gender": user.gender,
        "age": user.age,
        "role": user.role,
        "status": user.status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def require_login():
    user_id = session.get("user_id")
    if not user_id:
        return None, (jsonify({"message": "Please log in first."}), 401)
    user = db.session.get(User, user_id)
    if not user or user.status == "blocked":
        session.clear()
        return None, (jsonify({"message": "Your session is no longer valid."}), 401)
    return user, None


def require_admin():
    user, error = require_login()
    if error:
        return None, error
    if user.role != "admin":
        return None, (jsonify({"message": "Administrator access required."}), 403)
    return user, None


def predict_relevance(image):
    model = load_model("relevance")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
        predicted = torch.argmax(output, dim=1)
    result = predicted.item()
    unload_model()
    return result


def predict_tumor(image):
    model = load_model("classification")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
        predicted = torch.argmax(output, dim=1)
    result = predicted.item()
    unload_model()
    return result


def predict_segmentation(image):
    model = load_model("segmentation")
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)
        mask = (probability > 0.5).float()

    result = mask.squeeze().cpu().numpy()
    unload_model()
    return result


def mask_to_data_url(mask):
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.get("/api/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        database = "connected"
    except Exception:
        database = "error"
    return jsonify({"status": "ok", "database": database, "device": str(device)})


@app.get("/api/auth/me")
def me():
    user, error = require_login()
    if error:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": user_payload(user)})


@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    required = ["name", "email", "password", "confirm_password", "age", "gender", "mobile"]
    if any(not str(data.get(key, "")).strip() for key in required):
        return jsonify({"message": "All registration fields are required."}), 400

    name = str(data["name"]).strip()
    email = str(data["email"]).strip().lower()
    password = str(data["password"])
    confirm_password = str(data["confirm_password"])
    mobile = str(data["mobile"]).strip()

    try:
        age = int(data["age"])
    except (TypeError, ValueError):
        return jsonify({"message": "Age must be a number."}), 400

    gender = str(data["gender"]).upper()
    if gender not in {"M", "F", "O"}:
        return jsonify({"message": "Invalid gender."}), 400
    if len(mobile) != 10 or not mobile.isdigit():
        return jsonify({"message": "Mobile number must be exactly 10 digits."}), 400
    if password != confirm_password:
        return jsonify({"message": "Passwords do not match."}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "Email address is already registered."}), 409
    if User.query.filter_by(name=name).first():
        return jsonify({"message": "Name is already taken."}), 409

    requested_role = str(data.get("role", "user")).lower()
    allow_admin = os.getenv("ALLOW_ADMIN_REGISTRATION", "false").lower() == "true"
    role = "admin" if requested_role == "admin" and allow_admin else "user"
    status = "pending" if role == "admin" else "active"

    user = User(
        name=name,
        email=email,
        password=bcrypt.generate_password_hash(password).decode("utf-8"),
        mobile=mobile,
        gender=gender,
        age=age,
        role=role,
        status=status,
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "Registration successful. You can now log in.",
        "user": user_payload(user),
    }), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    role = str(data.get("role", "user")).lower()

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Invalid email or password."}), 401

    if user.role != role:
        return jsonify({"message": f"Account exists, but not as an {role}."}), 403
    if user.status == "blocked":
        return jsonify({"message": "Your account has been blocked."}), 403
    if user.status == "pending":
        return jsonify({"message": "Your admin account is awaiting approval."}), 403

    session.clear()
    session["user_id"] = user.id
    session["role"] = user.role

    return jsonify({"message": "Login successful.", "user": user_payload(user)})


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully."})


@app.post("/api/predict")
def predict():
    _, error = require_login()
    if error:
        return error

    if "image" not in request.files:
        return jsonify({"message": "No image was uploaded."}), 400

    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"message": "Please choose an image."}), 400

    allowed = {".png", ".jpg", ".jpeg", ".gif"}
    extension = Path(uploaded.filename).suffix.lower()
    if extension not in allowed:
        return jsonify({"message": "Allowed files: PNG, JPG, JPEG, GIF."}), 400

    temp_path = BASE_DIR / f"_upload_{uuid.uuid4().hex}{extension}"
    uploaded.save(temp_path)

    try:
        image = Image.open(temp_path).convert("RGB")

        relevance = predict_relevance(image)
        if relevance == 1:
            return jsonify({
                "status": "irrelevant",
                "message": "The uploaded image is irrelevant. Please upload a brain MRI image.",
            })

        tumor = predict_tumor(image)
        if tumor != 1:
            return jsonify({
                "status": "no_tumor",
                "message": "No tumor detected in the image.",
            })

        mask = predict_segmentation(image)
        return jsonify({
            "status": "tumor",
            "message": "Tumor detected.",
            "mask": mask_to_data_url(mask),
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"message": f"Prediction failed: {exc}"}), 500
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/admin/users")
def admin_users():
    _, error = require_admin()
    if error:
        return error
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({"users": [user_payload(user) for user in users]})


@app.post("/api/admin/users/<int:user_id>/block")
def block_user(user_id):
    current_admin, error = require_admin()
    if error:
        return error
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    if user.id == current_admin.id:
        return jsonify({"message": "You cannot block your own admin account."}), 400
    user.status = "blocked"
    db.session.commit()
    return jsonify({"message": "User blocked successfully."})


@app.post("/api/admin/users/<int:user_id>/unblock")
def unblock_user(user_id):
    _, error = require_admin()
    if error:
        return error
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    user.status = "active"
    db.session.commit()
    return jsonify({"message": "User unblocked successfully."})


@app.delete("/api/admin/users/<int:user_id>")
def delete_user(user_id):
    current_admin, error = require_admin()
    if error:
        return error
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    if user.id == current_admin.id:
        return jsonify({"message": "You cannot delete your own admin account."}), 400
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted successfully."})


@app.post("/api/admin/users/<int:user_id>/approve")
def approve_user(user_id):
    _, error = require_admin()
    if error:
        return error
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    user.status = "active"
    db.session.commit()
    return jsonify({"message": "Admin approved successfully."})


@app.post("/api/admin/profile/password")
def change_admin_password():
    admin, error = require_admin()
    if error:
        return error

    data = request.get_json(silent=True) or {}
    current_password = str(data.get("current_password", ""))
    new_password = str(data.get("new_password", ""))
    confirm_password = str(data.get("confirm_password", ""))

    if not bcrypt.check_password_hash(admin.password, current_password):
        return jsonify({"message": "Incorrect current password."}), 400
    if len(new_password) < 8:
        return jsonify({"message": "New password must be at least 8 characters."}), 400
    if new_password != confirm_password:
        return jsonify({"message": "New passwords do not match."}), 400

    admin.password = bcrypt.generate_password_hash(new_password).decode("utf-8")
    db.session.commit()
    return jsonify({"message": "Password changed successfully."})


def seed_admin():
    email = os.getenv("ADMIN_EMAIL", "admin@lifecare.com").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "").strip()
    if not password:
        print("ADMIN_PASSWORD is not set; no default admin will be created.")
        return

    admin = User.query.filter_by(email=email).first()
    if admin:
        return

    admin = User(
        name=os.getenv("ADMIN_NAME", "LifeCare Admin"),
        email=email,
        password=bcrypt.generate_password_hash(password).decode("utf-8"),
        mobile=os.getenv("ADMIN_MOBILE", "0000000000"),
        gender=os.getenv("ADMIN_GENDER", "O"),
        age=int(os.getenv("ADMIN_AGE", "30")),
        role="admin",
        status="active",
    )
    db.session.add(admin)
    db.session.commit()
    print(f"Created initial admin account: {email}")


with app.app_context():
    db.create_all()
    seed_admin()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=False)
