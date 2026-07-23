# 🧠 Brain Tumour Classification & Segmentation using AI

**An AI-powered full-stack web application that automates brain tumour detection and segmentation from MRI scans using Deep Learning, Flask, and PyTorch.**

🔗 **Live Demo:** {{[LIVE_URL](https://brain-tumour-classification-using-ai-production.up.railway.app/)}}
📂 **Source Code:** {{[GITHUB_REPO](https://github.com/ArigalaPunithKumar/Brain-Tumour-Classification-using-AI)}}

---

## 📖 About

Brain Tumour Classification & Segmentation using AI is a full-stack healthcare application designed to assist in the early detection of brain tumours from MRI scans. The platform combines modern web technologies with deep learning models to provide fast and accurate diagnostic support.

The application follows a three-stage AI pipeline that first verifies whether the uploaded image is a brain MRI, then classifies the presence of a tumour, and finally generates a precise segmentation mask highlighting the tumour region. Built with **Flask**, **PyTorch**, and **SQLite**, the system offers a secure and user-friendly interface for medical image analysis.

---

## ✨ Features

### 👤 User Authentication

* Secure user registration and login
* Password encryption using Flask-Bcrypt
* Session management
* SQLite database for user data

### 🧠 AI-Based Brain MRI Analysis

* Automatic brain MRI verification
* Brain tumour classification
* Tumour segmentation with visual mask generation
* Instant prediction results

### 🖼 Medical Image Processing

* Upload MRI scans (JPG/PNG)
* Image preprocessing and validation
* Segmented tumour visualization
* Downloadable prediction results

### 🌐 Web Application

* Responsive user interface
* Secure image upload
* Dynamic result visualization
* User-friendly workflow

---

## 🤖 AI Pipeline

### Stage 1 – MRI Relevance Detection

A fine-tuned **MobileNet** model verifies whether the uploaded image is a valid brain MRI before further processing.

### Stage 2 – Brain Tumour Classification

A second **MobileNet** model classifies the MRI as either:

* Tumour Detected
* Healthy (No Tumour)

### Stage 3 – Tumour Segmentation

If a tumour is detected, a **U-Net model with a ResNet34 encoder** generates an accurate segmentation mask highlighting the tumour boundaries.

---

## 🛠 Tech Stack

### Frontend

* HTML5
* CSS3
* Jinja2 Templates

### Backend

* Python
* Flask
* Flask-SQLAlchemy
* Flask-Bcrypt

### Database

* SQLite

### Artificial Intelligence

* PyTorch
* TorchVision
* segmentation-models-pytorch
* MobileNet
* U-Net
* ResNet34

### Image Processing

* OpenCV
* Pillow
* NumPy
* Matplotlib

### Deployment

* Docker
* Gunicorn
* Render / Railway Ready

---

## 🏗 Project Architecture

* Monolithic Flask Architecture
* Secure User Authentication
* Deep Learning Inference Pipeline
* Medical Image Processing
* SQLite Database Management
* AI Model Integration
* Responsive Web Interface

---

## 🚀 Key Highlights

* Developed a full-stack medical imaging application for automated brain tumour diagnosis.
* Built a secure authentication system with encrypted passwords and session management.
* Implemented a three-stage AI pipeline for MRI validation, tumour classification, and tumour segmentation.
* Integrated multiple deep learning models into a single Flask application.
* Generated accurate tumour segmentation masks using a U-Net architecture with a ResNet34 encoder.
* Processed MRI images using OpenCV, Pillow, and NumPy for efficient inference.
* Designed an intuitive web interface for uploading MRI scans and visualizing AI predictions.
* Prepared the application for cloud deployment using Docker and Gunicorn.

---

## 📂 Project Modules

* User Authentication
* User Management
* MRI Image Upload
* MRI Validation Model
* Brain Tumour Classification
* Tumour Segmentation
* Image Processing
* Prediction Visualization
* Database Management
* AI Model Integration

---

## 📚 What I Learned

* Building AI-powered healthcare applications using Flask and PyTorch.
* Integrating multiple deep learning models into a production-ready web application.
* Processing medical images with OpenCV and Pillow.
* Implementing secure authentication and database management.
* Deploying machine learning applications using Docker and Gunicorn.
* Designing complete AI inference pipelines for real-world medical applications.

---

## ❓ Frequently Asked Questions

### What is Brain Tumour Classification & Segmentation using AI?

It is a full-stack web application that uses Deep Learning to automatically detect and segment brain tumours from MRI scans, providing fast and accurate diagnostic assistance.

### Which technologies were used?

* **Frontend:** HTML5, CSS3, Jinja2
* **Backend:** Flask, Python
* **Database:** SQLite
* **AI:** PyTorch, MobileNet, U-Net, ResNet34
* **Image Processing:** OpenCV, Pillow, NumPy

### How does the AI pipeline work?

The application processes MRI scans in three stages:

1. **MRI Validation** – Confirms the uploaded image is a brain MRI.
2. **Tumour Classification** – Detects whether a tumour is present.
3. **Tumour Segmentation** – Generates a visual mask outlining the tumour region.

### What image formats are supported?

The application accepts common medical image formats such as **JPG** and **PNG**.

### Is user data secure?

Yes. User passwords are securely hashed using **Flask-Bcrypt**, and authentication is managed through secure user sessions.

### Can the application be deployed to the cloud?

Yes. The project is Dockerized and configured for deployment on platforms such as **Render** and **Railway**.

---

## ⭐ Conclusion

Brain Tumour Classification & Segmentation using AI is a comprehensive AI-powered healthcare application that combines secure web technologies with advanced deep learning models to deliver automated MRI validation, brain tumour detection, and precise tumour segmentation through an intuitive and scalable web platform.
