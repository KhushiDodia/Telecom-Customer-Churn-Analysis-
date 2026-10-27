from flask import Flask, render_template, request, jsonify
import pickle          
import numpy as np     
import pandas as pd    


app = Flask(__name__)

model = pickle.load(open('churn_model.pkl','rb'))   
scaler = pickle.load(open('scaler.pkl','rb'))   
columns = pickle.load(open('columns.pkl','rb'))   

@app.route('/')                        # URL: http://localhost:5000/
def home():
    """Show the main prediction page"""
    return render_template('index.html')
    


@app.route('/predict', methods=['POST'])   # URL: /predict  (only accepts POST)
def predict():
    
    data = request.form
    input_data = {col: 0 for col in columns}

    input_data['tenure'] = float(data.get('tenure', 0))
    input_data['MonthlyCharges'] = float(data.get('MonthlyCharges', 0))
    input_data['TotalCharges'] = float(data.get('TotalCharges', 0))
    input_data['SeniorCitizen'] = int(data.get('SeniorCitizen', 0))

    
    def set_if(col, value, field):
        """Helper: set input_data[col] = 1 if form field equals value"""
        if data.get(field) == value and col in input_data:
            input_data[col] = 1

    set_if('gender_Male', 'Male', 'gender')
    set_if('Partner_Yes', 'Yes', 'Partner')
    set_if('Dependents_Yes', 'Yes', 'Dependents')
    set_if('PhoneService_Yes','Yes', 'PhoneService')
    set_if('MultipleLines_No phone service', 'No phone service', 'MultipleLines')
    set_if('MultipleLines_Yes', 'Yes', 'MultipleLines')
    set_if('InternetService_Fiber optic', 'Fiber optic', 'InternetService')
    set_if('InternetService_No', 'No', 'InternetService')
    set_if('OnlineSecurity_No internet service','No internet service', 'OnlineSecurity')
    set_if('OnlineSecurity_Yes', 'Yes', 'OnlineSecurity')
    set_if('OnlineBackup_No internet service', 'No internet service', 'OnlineBackup')
    set_if('OnlineBackup_Yes', 'Yes', 'OnlineBackup')
    set_if('DeviceProtection_No internet service','No internet service', 'DeviceProtection')
    set_if('DeviceProtection_Yes', 'Yes', 'DeviceProtection')
    set_if('TechSupport_No internet service', 'No internet service', 'TechSupport')
    set_if('TechSupport_Yes', 'Yes', 'TechSupport')
    set_if('StreamingTV_No internet service', 'No internet service', 'StreamingTV')
    set_if('StreamingTV_Yes', 'Yes', 'StreamingTV')
    set_if('StreamingMovies_No internet service','No internet service', 'StreamingMovies')
    set_if('StreamingMovies_Yes', 'Yes', 'StreamingMovies')
    set_if('Contract_One year', 'One year', 'Contract')
    set_if('Contract_Two year', 'Two year', 'Contract')
    set_if('PaperlessBilling_Yes', 'Yes', 'PaperlessBilling')
    set_if('PaymentMethod_Credit card (automatic)', 'Credit card (automatic)', 'PaymentMethod')
    set_if('PaymentMethod_Electronic check', 'Electronic check', 'PaymentMethod')
    set_if('PaymentMethod_Mailed check', 'Mailed check', 'PaymentMethod')

    
    X = pd.DataFrame([input_data])[columns]

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]          # 0 = Stay, 1 = Churn
    probability = model.predict_proba(X_scaled)[0]   # [prob_stay, prob_churn]

    churn_prob = round(float(probability[1]) * 100, 2)
    stay_prob  = round(float(probability[0]) * 100, 2)

    
    if churn_prob >= 70:
        risk = "High Risk"
    elif churn_prob >= 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    
    tips = []
    if data.get('Contract') == 'Month-to-month':
        tips.append("Offer discount for switching to a 1 or 2-year contract")
    if float(data.get('MonthlyCharges', 0)) > 70:
        tips.append("Monthly charges are high — offer a loyalty discount")
    if float(data.get('tenure', 0)) < 12:
        tips.append("New customer — assign dedicated onboarding support")
    if data.get('TechSupport') == 'No':
        tips.append("Offer free Tech Support upgrade for 3 months")
    if data.get('OnlineSecurity') == 'No':
        tips.append("Offer Online Security as a free add-on")
    if data.get('PaymentMethod') == 'Electronic check':
        tips.append("Encourage switching to automatic payment method")
    if not tips:
        tips.append("Send a satisfaction survey and loyalty reward offer")

    #Send Result Back to Frontend 
    return jsonify({
        'prediction':  int(prediction),     # 0 or 1
        'churn_prob':  churn_prob,           # e.g. 73.5
        'stay_prob':   stay_prob,            # e.g. 26.5
        'risk':        risk,                 # "High Risk" etc.
        'tips':        tips                  # list of recommendations
    })
    


# Run the App 
if __name__ == '__main__':
    app.run(debug=True)
    

