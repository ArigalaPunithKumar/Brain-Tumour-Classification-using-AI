# Brain Tumour Classification & Segmentation using AI

An AI-powered web application that analyzes brain MRI images through a multi-stage deep-learning pipeline for MRI validation, tumour classification, and tumour-region segmentation.

> Academic/software-development project. It is not intended to replace professional medical diagnosis.

## Problem

This project explores how computer vision and deep learning can be integrated into a web application to assist with MRI image analysis and visualize predicted tumour regions.

## AI Pipeline

```text
MRI Image
   |
   v
MRI Relevance Detection
   |
   v
Tumour Classification
   |
   v
Tumour Segmentation
   |
   v
Prediction + Visual Mask
```

### Stage 1 — MRI Relevance Detection
A fine-tuned MobileNet model verifies whether the uploaded image is a relevant brain MRI.

### Stage 2 — Tumour Classification
A MobileNet-based classifier predicts whether a tumour is detected.

### Stage 3 — Tumour Segmentation
A U-Net model with a ResNet34 encoder generates a mask representing the predicted tumour region.

## Key Features

- User registration and login
- Password hashing with Flask-Bcrypt
- Session-based authentication
- MRI image upload
- MRI relevance verification
- Brain tumour classification
- Tumour segmentation
- Segmentation-mask visualization
- Aiven MySQL database integration
- Responsive web interface
- Docker/Gunicorn deployment configuration

## Technology Stack

**Application:** React, Vite, Python, Flask, Flask-SQLAlchemy, Flask-Bcrypt, Aiven MySQL

**Deep Learning:** PyTorch, TorchVision, segmentation-models-pytorch, MobileNet, U-Net, ResNet34

**Image Processing:** OpenCV, Pillow, NumPy, Matplotlib

**Deployment:** Vercel (frontend), Render (backend), Gunicorn, Aiven MySQL

## My Contributions

- Developed the web application around the deep-learning inference workflow.
- Integrated image upload and preprocessing.
- Connected classification and segmentation models with Flask.
- Implemented authentication and database-backed user management.
- Built prediction and segmentation-mask visualization.
- Worked on Docker/Gunicorn deployment configuration.

## Project Flow

```text
Authentication
     |
     v
MRI Upload
     |
     v
Preprocessing
     |
     v
MRI Validation
     |
     v
Classification
     |
     v
Segmentation
     |
     v
Result Visualization
```

## Deployment

- `frontend/` is deployed to Vercel.
- `backend/` is deployed to Render.
- Aiven MySQL stores user accounts and application data.
- The three production model files are committed under `backend/models/`.

See `DEPLOYMENT.md` for the complete deployment configuration and environment variables.

## Deployment

The production setup uses Vercel for the React frontend, Render for the Flask/PyTorch backend, and Aiven MySQL for persistent data. Configure the Vercel and Render URLs using the steps in `DEPLOYMENT.md`.

## What This Project Demonstrates

- Deep-learning model integration
- Computer-vision preprocessing
- Flask development
- Model inference pipelines
- Image segmentation
- Database integration
- Authentication
- Deployment-oriented development

## Future Improvements

- Expanded and balanced datasets
- More comprehensive model evaluation
- Improved inference performance
- Automated tests
- Model versioning
- Better prediction explainability

## Author

**A Punith Kumar**

[GitHub](https://github.com/ArigalaPunithKumar)
