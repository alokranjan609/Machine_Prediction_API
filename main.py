from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Initialize FastAPI
app = FastAPI()

# Global variables
DATASET_PATH = "uploaded_dataset.csv"
MODEL_PATH = "downtime_predictor.pkl"
SCALER_PATH = "scaler.pkl"
model = None

# Pydantic model for input validation
class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float


def preprocess_data(data):
    """
    Preprocess the data by handling missing values, duplicates, outliers, and scaling features.
    """
    # Validate required columns
    required_columns = ['Temperature', 'Run_Time', 'Downtime_Flag']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {', '.join(required_columns)}")

    # Handle missing values
    if data.isnull().sum().sum() > 0:
        print("Handling missing values...")
        data = data.dropna(subset=['Downtime_Flag'])  # Ensure target is not missing
        data['Temperature'] = data['Temperature'].fillna(data['Temperature'].median())
        data['Run_Time'] = data['Run_Time'].fillna(data['Run_Time'].median())

    # Remove duplicate rows
    if data.duplicated().sum() > 0:
        print("Removing duplicate rows...")
        data = data.drop_duplicates()

    # Handle outliers using IQR
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print("Handling outliers...")
    data = remove_outliers(data, 'Temperature')
    data = remove_outliers(data, 'Run_Time')

    # Ensure correct data types
    print("Ensuring correct data types...")
    data['Temperature'] = data['Temperature'].astype(float)
    data['Run_Time'] = data['Run_Time'].astype(float)
    data['Downtime_Flag'] = data['Downtime_Flag'].astype(int)

    # Extract features and target
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']

    # Feature scaling using StandardScaler
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler





@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload a CSV file for training.
    """
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    try:
        contents = await file.read()
        with open(DATASET_PATH, "wb") as f:
            f.write(contents)
        return {"message": "File uploaded successfully.", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/train")
async def train_model():
    """
    Train the model on the uploaded dataset with advanced preprocessing.
    """
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    try:
        # Load dataset
        data = pd.read_csv(DATASET_PATH)
        
        # Preprocess data
        X, y, scaler = preprocess_data(data)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        global model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model and scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        return {
            "message": "Model trained successfully with preprocessing.",
            "accuracy": accuracy,
            "classification_report": metrics
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")



@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Make a prediction using the trained model with preprocessing.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    if not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Scaler not found. Please retrain the model.")
    
    try:
        # Load model and scaler
        global model
        if model is None:
            model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data.dict()])
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        prediction = model.predict(input_scaled)
        confidence = model.predict_proba(input_scaled).max(axis=1)[0]
        
        downtime = "Yes" if prediction[0] == 1 else "No"
        return {"Downtime": downtime, "Confidence": round(confidence, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
