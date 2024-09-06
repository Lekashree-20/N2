from flask import Flask, request, jsonify
import pandas as pd
import pickle
import google.generativeai as gemini

import google.generativeai as genai
from flask_cors import CORS

# Configure the Gemma API
genai.configure(api_key="AIzaSyD1FPKl0lENNaIw8JGtMBzPXopVDIqcab8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the models and scaler
with open('models/ensemble_lin_rbf.pkl', 'rb') as file:
    ensemble_lin_rbf = pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Combined endpoint for stroke prediction and report generation
@app.route('/predict_and_generate_report', methods=['POST'])
def predict_and_generate_report():
    try:
        # Get input data from the request
        data = request.json
        
        # Define expected features based on training data
        feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                         'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        
        # Check if all required features are provided
        if not all(feature in data for feature in feature_names):
            return jsonify({"error": "Missing one or more required features."}), 400

        # Extract values and create a DataFrame
        input_data = [data[feature] for feature in feature_names]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Apply scaling
        std_data = scaler.transform(input_df)
        
        # Predict using ensemble model
        prediction = ensemble_lin_rbf.predict(std_data)
        prediction_proba = ensemble_lin_rbf.predict_proba(std_data)[0][1]  # Probability of stroke
        risk_name="Neurology"
        # Generate prediction result and risk score
        predicted_label = 'The patient had a stroke' if prediction[0] == 1 else 'The patient did not have a stroke'
        
        # Initialize response
        response = {
            "risk":risk_name,
            "prediction": predicted_label,
            "risk_score": prediction_proba
        }

        # If the patient had a stroke, generate the prevention report
        if predicted_label == 'The patient had a stroke':
            age = data.get('age', 30)  # Default age if not provided explicitly
            risk = "Stroke"
            disease = "Stroke"
            
            report = generate_prevention_report(risk, disease, age)
            if report:
                response["report"] = report
            else:
                response["report"] = "Failed to generate a report."

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to generate prevention report using Gemma API
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
       - Purpose of the report
       - Context of general health and wellness

    2. **Risk Description**
       - General description of the identified risk
       - Common factors associated with the risk

    3. **Stage of Risk**
       - General information about the risk stage
       - Any typical considerations

    4. **Risk Assessment**
       - General overview of the risk's impact on health

    5. **Findings**
       - General wellness observations
       - Supporting information

    6. **Recommendations**
       - General wellness tips and lifestyle changes
       - Actions to promote well-being

    7. **Way Forward**
       - Suggested next steps for maintaining health
       - Advanced follow-up actions on what to do for this risk and how to overcome it.

    8. **Conclusion**
       - Summary of overall wellness advice
       - General support resources

    9. **Contact Information**
       - Information for general inquiries

    10. **References**
        - Simplified wellness resources (if applicable)

    **Details:**
    Risk: {risk}
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
