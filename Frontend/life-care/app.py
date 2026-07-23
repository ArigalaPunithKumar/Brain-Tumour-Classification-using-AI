from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os 
from datetime import datetime
from functools import wraps
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import torch.nn as nn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

db_url = os.environ.get('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///users.db'

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# Ensure the upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations for classification and relevancy
image_transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the MobileNet model class (same as the one used during training)
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the trained classification model for tumor/no-tumor detection
classification_model = MobileNetModel(num_classes=2)
classification_model.load_state_dict(torch.load("mobilenet.pt", map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()

# Load the trained relevancy detection model
relevancy_model = MobileNetModel(num_classes=2)
relevancy_model.load_state_dict(torch.load("mobilenet_irrelevent.pt", map_location=device))
relevancy_model = relevancy_model.to(device)
relevancy_model.eval()

# Load the segmentation model for tumor detection
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
segmentation_model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=1,  # Input channels for grayscale images
    classes=1,      # Output classes
    activation=None
)
segmentation_model.load_state_dict(torch.load('best_model.pth', map_location=device))
segmentation_model.to(device)
segmentation_model.eval()

# Predict image relevance
def predict_relevance(image):
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = relevancy_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Tumor detection in relevant images
def predict_image(image):
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = classification_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Preprocess image for segmentation
def preprocess_image_for_segmentation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Add channel dimension (1 channel for grayscale)
    return torch.tensor(image).to(device)

# Predict tumor segmentation
def predict_segmentation(image_path):
    input_image = preprocess_image_for_segmentation(image_path)

    with torch.no_grad():  # Disable gradient calculation
        output_logits = segmentation_model(input_image)

    output_prob = torch.sigmoid(output_logits)
    prediction = (output_prob > 0.5).float()  # Convert to binary mask

    return prediction.squeeze().cpu().numpy()  # Move back to CPU and remove unnecessary dimensions

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.Enum('M', 'F', 'O'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    role = db.Column(db.String(10), default='user')
    status = db.Column(db.String(10), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()  # Create the database tables
    if not User.query.filter_by(email='admin@lifecare.com').first():
        hashed_pw = bcrypt.generate_password_hash('admin123')
        admin_user = User(name='Admin', email='admin@lifecare.com', password=hashed_pw, mobile='0000000000', gender='M', age=30, role='admin', status='active')
        db.session.add(admin_user)
        db.session.commit()


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('admin')) if session.get('role') == 'admin' else redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'user')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            if user.role != role:
                flash(f'Account exists, but not as an {role}. Please select the correct role.', 'danger')
                return render_template('login.html')
            
            if user.status == 'blocked':
                flash('Your account has been blocked by the administrator.', 'danger')
                return render_template('login.html')
            elif user.status == 'pending':
                flash('Your admin account is still pending approval from an existing administrator.', 'warning')
                return render_template('login.html')

            session['user_id'] = user.id  # Store user ID in session
            session['role'] = user.role
            
            if user.role == 'admin':
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('admin')) if session.get('role') == 'admin' else redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name')   # Updated from 'username' to 'name'
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')
        role = request.form.get('role', 'user')

        # Validate mobile number
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('login.html')

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('login.html')

        # Check if name (username) already exists
        if User.query.filter_by(name=name).first():
            flash('Name is already taken. Please choose a different one.', 'danger')
            return render_template('login.html')

        # Validate password
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('login.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('login.html')

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password)

        # Determine status
        status = 'pending' if role == 'admin' else 'active'

        # Create a new user instance
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            age=age,
            gender=gender,
            mobile=mobile,
            role=role,
            status=status
        )

        # Add and commit the new user to the database
        db.session.add(new_user)
        db.session.commit()

        if role == 'admin':
            flash('Admin registration successful! Please wait for an existing administrator to approve your account.', 'success')
        else:
            flash('Registration successful! You can now log in.', 'success')
            
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')


#@app.route('/prediction', methods=['GET', 'POST'])
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Open image with PIL for relevancy and classification prediction
            image_pil = Image.open(image_path).convert('RGB')

            # Step 1: Check if the image is relevant
            relevance_prediction = predict_relevance(image_pil)
            result = {}
            result['image_filename'] = 'uploads/' + filename

            if relevance_prediction == 1:  # Irrelevant image
                result['message'] = "The uploaded image is irrelevant. Please upload Brain MRI image."
                return render_template('prediction.html', result=result)
            else:  # Relevant image, proceed with tumor prediction
                # Step 2: Classify the image for tumor presence
                tumor_prediction = predict_image(image_pil)
                if tumor_prediction == 1:  # Tumor detected
                    result['message'] = "Tumor detected."

                    # Perform segmentation
                    predicted_mask = predict_segmentation(image_path)

                    # Save the predicted mask
                    mask_filename = 'mask_' + filename
                    mask_path = os.path.join(app.config['OUTPUT_FOLDER'], mask_filename)
                    plt.imsave(mask_path, predicted_mask, cmap='gray')

                    result['mask_filename'] = 'outputs/' + mask_filename

                    return render_template('prediction.html', result=result)
                else:
                    result['message'] = "No tumor detected in the image."
                    return render_template('prediction.html', result=result)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif', 'danger')
            return redirect(request.url)
    else:
        return render_template('prediction.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@admin_required
def admin():
    users = User.query.order_by(User.created_at.desc()).all()
    total_users = User.query.filter_by(role='user').count()
    pending_admins = User.query.filter_by(role='admin', status='pending').count()
    blocked_users = User.query.filter_by(status='blocked').count()
    return render_template('admin.html', users=users, total_users=total_users, pending_admins=pending_admins, blocked_users=blocked_users)

@app.route('/admin/block/<int:user_id>')
@admin_required
def block_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role != 'admin' or user.id != session.get('user_id'):
        user.status = 'blocked'
        db.session.commit()
        flash(f'User {user.email} has been blocked.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/unblock/<int:user_id>')
@admin_required
def unblock_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role != 'admin' or user.id != session.get('user_id'):
        user.status = 'active'
        db.session.commit()
        flash(f'User {user.email} has been unblocked.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/delete/<int:user_id>')
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role != 'admin' or user.id != session.get('user_id'):
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.email} has been deleted.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/approve/<int:user_id>')
@admin_required
def approve_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.status == 'pending':
        user.status = 'active'
        db.session.commit()
        flash(f'Admin {user.email} has been approved.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/edit/<int:user_id>', methods=['POST'])
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    # Allow editing normal users, or if it's the admin themselves
    if user.role != 'admin' or user.id == session.get('user_id'):
        user.name = request.form.get('name', user.name)
        user.email = request.form.get('email', user.email)
        user.mobile = request.form.get('mobile', user.mobile)
        user.age = request.form.get('age', user.age)
        user.gender = request.form.get('gender', user.gender)
        db.session.commit()
        flash(f'User {user.email} updated successfully.', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/profile', methods=['POST'])
@admin_required
def admin_profile():
    admin_id = session.get('user_id')
    admin_user = User.query.get(admin_id)
    
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not bcrypt.check_password_hash(admin_user.password, current_password):
        flash('Incorrect current password.', 'danger')
        return redirect(url_for('admin'))
        
    if new_password != confirm_password:
        flash('New passwords do not match.', 'danger')
        return redirect(url_for('admin'))
        
    admin_user.password = bcrypt.generate_password_hash(new_password)
    db.session.commit()
    flash('Password changed successfully.', 'success')
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)