# Smart Healthcare Assistant

## Project Overview
The **Smart Healthcare Assistant** is an intelligent web application designed to provide **early risk assessment** for major diseases, including **Diabetes, Heart Disease, Liver Disease, and Kidney Disease**.  
It leverages **machine learning models** to deliver predictions, health tips, and specialist recommendations based on either **user-provided lab values** or **selected symptoms**.  

This project is aimed at helping users **understand potential health risks** and encouraging them to seek professional medical advice.  
It also serves as a **comprehensive demo for AI-driven healthcare applications**.

---

## Key Features

- **Disease Prediction**
  - Supports Diabetes, Heart Disease, Liver Disease, and Kidney Disease.
  - Two input modes:
    - **Lab Reports (exact values)**
    - **Symptoms (approximate values)** — generates estimated lab values.
  - Provides **risk classification** (High Risk / Low Risk) and **probability estimates**.

- **History Tracking**
  - Records all predictions with timestamp, inputs, symptoms, and model outputs.
  - Users can **view, export (CSV), or clear history**.

- **Health Tips & Recommendations**
  - Displays personalized health tips based on prediction.
  - Suggests **appropriate specialists** (Endocrinologist, Cardiologist, Hepatologist, Nephrologist).

- **Chatbot Assistance**
  - General health guidance via rule-based responses.
  - Optional integration with **GPT4All** for conversational support.
  - Supports **translation** into multiple languages using Deep Translator.

- **CSV Upload**
  - Upload CSV files with lab results; app uses the **first row** for predictions.

- **PDF Reports**
  - Generate downloadable PDF reports summarizing prediction results and recommendations (requires ReportLab).

- **Interactive Plots**
  - Visualizes lab values vs normal ranges for key parameters using Matplotlib.

---

## Technical Stack

- **Frontend & UI:** Streamlit  
- **Machine Learning:** Scikit-learn (`.pkl` models)  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **PDF Generation:** ReportLab (optional)  
- **Translation:** Deep Translator (optional)  
- **Chatbot:** GPT4All (optional)  
- **Deployment:** Streamlit Cloud (or alternative hosting platforms)

---

## How It Works

1. User selects a **disease** and **input mode**.  
2. Inputs are collected and **preprocessed**.  
3. Corresponding **ML model is loaded on demand**.  
4. Model predicts **risk level** and **probability**.  
5. Rule-based overrides adjust predictions for severe symptom combinations.  
6. App displays:
   - Risk status  
   - Health tips  
   - Specialist recommendations  
   - Optional visualizations  
7. Prediction results are **saved in history**, and users can **download PDF reports** or **ask the chatbot**.

---



