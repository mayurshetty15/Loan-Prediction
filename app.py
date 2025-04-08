from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pandas as pd
app = Flask(__name__)
import joblib
model_files = {
    'knn': 'models/knn',
    'decision_tree': 'models/dtree',
    'random_forest': 'models/rf',
    'gaussian_naive_bayes': 'models/gnb',
}

models = {}
scalers = {}
encoders = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type')
        print(f"Using model type: {model_type}")
        
        features = {
            'no_of_dependents': int(data.get('no_of_dependents')),
            'education': data.get('education'),
            'self_employed': data.get('self_employed'),
            'income_annum': float(data.get('income_annum')),
            'loan_amount': float(data.get('loan_amount')),
            'loan_term': float(data.get('loan_term')),
            'cibil_score': float(data.get('cibil_score')),
            'residential_assets_value': float(data.get('residential_assets_value')),
            'commercial_assets_value': float(data.get('commercial_assets_value')),
            'luxury_assets_value': float(data.get('luxury_assets_value')),
            'bank_asset_value': float(data.get('bank_asset_value'))
        }
        features['total_assets']=features['bank_asset_value']+features['commercial_assets_value']+features['residential_assets_value']+features['luxury_assets_value']
        df = pd.DataFrame([features])
        
        categorical_cols = ['education', 'self_employed']
        model=models[model_type]
        encoder=encoders[model_type]
        scaler=scalers[model_type]
        le = encoder
        for col in categorical_cols:
            df[col] = le[col].transform(df[col].astype(str))
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        prediction_proba = None
        

            
        
        result = {
            'success': True,
            'prediction': le['loan_status'].inverse_transform(prediction)[0],
            'model_used': model_type
        }
        

            
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()  
        return jsonify({'error': str(e)})

@app.route('/model_info', methods=['GET'])
def model_info():
    model_info = {
        'knn': {
            'name': 'K-Nearest Neighbors',
            'description': 'Makes predictions based on similar loan applications in the dataset'
        },
        'decision_tree': {
            'name': 'Decision Tree',
            'description': 'Uses a tree-like model of decisions based on application features'
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'An ensemble of decision trees for higher accuracy and resistance to overfitting'
        },
        'gaussian_naive_bayes': {
            'name': 'Gaussian Naive Bayes',
            'description': 'Assumes continuous data follows a Gaussian (normal) distribution and applies Bayes theorem for classification'
        },

    }

    available_models = {}
    for model_key, filepath in model_files.items():
        available_models[model_key] = os.path.exists(filepath)
    
    return jsonify({
        'models': model_info,
        'available': available_models
    })

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path+'/model.pkl')
                scaler = joblib.load(model_path+'/scaler.pkl')
                label_encoders = joblib.load(model_path+'/label_encoders.pkl')
                models[model_name]=model
                scalers[model_name]=scaler
                encoders[model_name]=label_encoders
            except Exception as e:
                print(e)
                print(f"Error loading {model_name} model: {e}")
    
    app.run(debug=True)