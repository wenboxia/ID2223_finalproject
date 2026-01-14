# 🚤 Flores-Corvo Boat Forecaster (Captain Joao)
**ID2223 Scalable Machine Learning Final Project**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]
(https://f124b88f7dc56bb60e.gradio.live)

## 👤 Author Information
* **Name:** Wenbo Xia
* **Course:** ID2223 Scalable Machine Learning
* **Project Type:** Interactive AI System / Serverless ML

---

## 📋 Project Compliance Analysis
This project was designed to strictly adhere to the ID2223 Final Project requirements. Below is a summary of how each core requirement is met:

| Requirement | Status | Implementation Details |
| :--- | :--- | :--- |
| **1. Dynamic Data Source** | ✅ **Met** | The project does **not** use static datasets (e.g., Kaggle). Instead, it fetches **real-time and historical weather data** dynamically from the **Open-Meteo API**. |
| **2. Non-trivial Prediction** | ✅ **Met** | The problem is **operational decision-making**: Predicting whether high wind gusts (>30km/h) will force the cancellation of boat trips between Flores and Corvo islands. It uses a Gradient Boosting Regressor trained on multiple features (Temperature, Precipitation, Wind Direction). |
| **3. Cloud Storage (Full Stack)** | ✅ **Met** | The pipeline is serverless. **Hopsworks Feature Store** is used for data versioning and storage. **Hopsworks Model Registry** is used for model versioning and serving. |
| **4. User Interface (UI)** | ✅ **Met** | A graphical interface is provided via **Gradio** hosted on **Hugging Face Spaces**, allowing users to query boat status interactively. |
| **5. Interactive AI (Bonus)** | ✅ **Met** | The project implements an **LLM Persona ("Captain Joao")**. It uses a RAG-like approach where ML predictions are injected into the context of **Qwen-2.5-7B**, converting raw data into natural, humorous advice. |

---

## 📖 Project Overview
Tourists in the Azores often face uncertainty regarding the ferry connection between **Flores** and **Corvo**. The Atlantic Ocean is rough, and trips are frequently cancelled when wind gusts exceed safety limits.

This project builds an end-to-end **Batch Inference Pipeline** that:
1.  Downloads weather data daily.
2.  Predicts wind gusts for the next 7 days.
3.  Advises users on whether the boat will run via an AI Chatbot.

## 🏗 System Architecture

The project follows a Serverless MLOps architecture:

1.  **Feature Pipeline** (`1_feature_pipeline.ipynb`):
    * Connects to Open-Meteo API.
    * Cleans and backfills historical weather data.
    * Upserts features to the **Hopsworks Feature Group** (`azores_wind_data`).

2.  **Training Pipeline** (`2_training_pipeline.ipynb`):
    * Creates a Feature View in Hopsworks.
    * Trains a **HistGradientBoostingRegressor** (Scikit-Learn).
    * Evaluates metrics (RMSE) and registers the model to the **Hopsworks Model Registry**.

3.  **Inference Pipeline & UI** (`app.py`):
    * Runs on **Hugging Face Spaces**.
    * Downloads the trained model from Hopsworks.
    * Fetches the *live* 7-day weather forecast.
    * Generates predictions and passes them to the **LLM (Captain Joao)** for a conversational response.

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `1_feature_pipeline.ipynb` | **Data Ingestion**. Fetches historical data from Open-Meteo and authenticates with Hopsworks to store features. |
| `2_training_pipeline.ipynb` | **Model Training**. Handles feature selection, model training, evaluation, and saving the model to the cloud registry. |
| `3_batch_inference.ipynb` | **Offline Inference Test**. A local script to verify the model predicts correctly on new forecast data. |
| `app.py` | **Gradio Application**. The deployment script for Hugging Face. Contains the logic for the "Captain Joao" LLM persona. |
| `requirements.txt` | **Dependencies**. Libraries required to run the project (hopsworks, gradio, openmeteo_requests, etc.). |

## 🚀 Live Demo
Access the running application on Hugging Face Spaces here:
👉 **[https://f124b88f7dc56bb60e.gradio.live]**

## 🤖 The "Captain Joao" Persona (Interactive AI)
Instead of showing a boring table of wind speeds, this project uses **Context-Aware Generation**.

* **Input:** User asks "Is it safe to go tomorrow?"
* **System Logic:**
    1.  The app fetches the ML prediction (e.g., "Max Gusts: 45 km/h").
    2.  The app applies business logic (Threshold > 30 km/h = Cancelled).
    3.  The app prompts the LLM: *"You are a captain. The wind is 45km/h. Tell the user the boat is cancelled."*
* **Output:** "Ahoy! 🌊 The seas are angry tomorrow (45 km/h gusts). The boat stays in the harbor. Better grab a glass of wine instead! 🍷"

---
*Created by Wenbo Xia for the ID2223 Final Project.*
