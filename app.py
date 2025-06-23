from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('rf_acc_68.pkl')        # Load your trained model
scaler = joblib.load('normalizer.pkl')      # Load your trained scaler

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect input values from form
        vals = [float(x) for x in request.form.values()]
        input_data = scaler.transform([vals])   # Normalize the input
        prediction = model.predict(input_data)[0]

        result = "Positive for Cirrhosis" if prediction == 1 else "Negative for Cirrhosis"
        return render_template('index.html', prediction=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
