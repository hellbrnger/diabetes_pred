from flask import Flask, render_template, request
import numpy as np
import pickle


with open('model.pkl', 'rb') as model_file:
    model, scaler = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        input_data = [float(request.form[key]) for key in request.form.keys()]
        input_array = np.array(input_data).reshape(1, -1)

    
        scaled_input = scaler.transform(input_array)

    
        prediction = model.predict(scaled_input)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
