from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn import metrics

app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load('model.sav')
except:
    model = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded file
        file = request.files['file']
        if not file:
            return render_template('home.html', error="Please upload a file")
        
        # Read data
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return render_template('home.html', error="Unsupported file format")
        
        # Basic preprocessing
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data.dropna(inplace=True)
        
        # Convert Churn to binary
        if 'Churn' in data.columns:
            data['Churn'] = np.where(data['Churn'] == 'Yes', 1, 0)
        
        # Create tenure groups
        labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
        data['tenure_group'] = pd.cut(data['tenure'], range(1, 80, 12), right=False, labels=labels)
        
        # Drop unnecessary columns