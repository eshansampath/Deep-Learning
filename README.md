🛑 APEX Plate Detect – Deep Learning Number Plate Recognition System

AI-powered number plate detection & classification system designed for vehicle monitoring at NIBM.

🚀 Project Overview

APEX Plate Detect is a complete deep learning–based product that identifies whether a vehicle belongs to NIBM or Non-NIBM using number plate images.
The system includes:

A ResNet18 (Transfer Learning) model trained for classification

A FastAPI backend for real-time predictions

A web dashboard UI for uploading images and viewing results

An SQLite database to store detection logs

Excel export for reporting and analysis

This project was developed as part of Deep Learning Course Work 2 at NIBM.

🧠 Core Features

🔍 Number plate detection & classification

🖥️ Web dashboard (upload → predict → download logs)

⚡ Real-time inference through FastAPI

🗂️ SQLite database for structured logging

📊 Excel export of predictions

🔧 End-to-end AI pipeline (data → training → API → UI)


📊 Dataset

Public Sri Lankan Number Plate Dataset from Kaggle

Custom dataset collected inside NIBM premises

Labelled using Label Studio

Preprocessing steps:

Resize → 224×224

Normalization (ImageNet standards)

Augmentation (flip, rotate, brightness, noise)

🧪 Model Training

Architecture: ResNet18 (Frozen Backbone) + Custom FC Layer

Loss: CrossEntropyLoss (with class weights)

Optimizer: Adam (lr=0.0005)

Regularization:

WeightedRandomSampler

Dropout (0.5)

Early stopping (patience=20)

LR Scheduler (ReduceLROnPlateau)

Final accuracy: 92%

Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

⚙️ API Endpoints (FastAPI)
/predict

Input: Image file

Output: JSON → { class, confidence, inference_time }

/logs

Stores prediction history in SQLite

Supports Excel export

🖥️ User Interface

Simple web dashboard

Upload image → get predictions → download results

Built for security teams & admin staff at NIBM

🏗️ End-to-End Architecture
User Interface (Web Dashboard)
        ↓
 FastAPI Backend → ResNet18 Model → Prediction
        ↓
  SQLite Database → Excel Export

🔒 Security Considerations

No personal data stored

Secure logs via SQLite

API validated inputs

Local deployment only (no internet exposure)

📦 Future Improvements

Docker containerization

Cloud deployment (AWS / Azure)

Support for multiple cameras

Real-time video detection

Integration with MLOps workflow
