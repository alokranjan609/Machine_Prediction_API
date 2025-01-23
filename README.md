# Predictive Analysis API for Manufacturing Operations

This project implements a RESTful API for predicting machine downtime based on manufacturing data. It uses a Logistic Regression model and includes endpoints for uploading datasets, training the model, and making predictions.

---

## Features

1. **Upload CSV Files**: Accepts manufacturing data for model training.
2. **Preprocess Data**: Cleans and preprocesses data for better training:
   - Handles missing values, duplicates, and outliers.
   - Standardizes feature values for consistency.
3. **Train the Model**: Trains a Logistic Regression model on the uploaded dataset.
4. **Make Predictions**: Predicts whether a machine will experience downtime based on input parameters.

---

## Technologies Used

- **FastAPI**: Framework for building APIs.
- **Uvicorn**: ASGI server for running FastAPI.
- **scikit-learn**: For machine learning model building and evaluation.
- **pandas**: For data manipulation.
- **joblib**: For saving and loading the trained model and scaler.

---

## Endpoints

### **1. Upload Dataset**
- **URL**: `/upload` (POST)
- **Description**: Upload a CSV file containing manufacturing data.
- **Input**: Multipart form-data (`file` field with a CSV file)
- **Output**:
  ```json
  {
    "message": "File uploaded successfully.",
    "filename": "synthetic_manufacturing_data.csv"
  }

### **2. Train Model**
- **URL**: `/train` (POST)
- **Description**: Preprocess the uploaded dataset and train a Logistic Regression model.
- **Input**: None
- **Output:**
  - Accuracy of the trained model.
  - Classification metrics (precision, recall, F1-score).

### **3.Predict**
- **URL**: /predict (POST)
- **Description**: Predict machine downtime based on input parameters.
- **Input**: JSON payload with `Temperature` and `Run_Time.`
  ```json
  {
   "Temperature": 85,
   "Run_Time": 130
  }
- **Output**:
  - Downtime prediction (Yes/No).
  - Confidence score for the prediction.
  ```json
  {
   "Downtime": "Yes",
   "Confidence": 0.87
  }

# Installation and Setup
### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
### Steps
- **1. Clone The Repository**
```
git clone https://github.com/alokranjan609/Machine_Prediction_API.git
cd predictive-analysis-api
```
- **2. Create a Virtual Environment:**
```
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```
- **3. Install Dependencies:**
  ```
  pip install -r requirements.txt
  ```
- **4. Run the Application:**
  ```
  uvicorn main:app --reload
  ```
### File Structure
```
├── main.py               # FastAPI application
├── uploaded_dataset.csv  # Uploaded dataset (placeholder)
├── downtime_predictor.pkl  # Saved model (after training)
├── scaler.pkl            # Saved scaler (after training)
├── requirements.txt      # Python dependencies
```
### Dataset Requirements
- The uploaded dataset must have the following columns:
  - Temperature (float): Machine temperature in degrees Celsius.
  - Run_Time (float): Machine runtime in minutes.
  - Downtime_Flag (int): Binary indicator (1 if downtime occurred, 0 otherwise).
