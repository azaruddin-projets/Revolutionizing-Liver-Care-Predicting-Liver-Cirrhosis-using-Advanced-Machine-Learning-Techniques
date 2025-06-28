from flask import Flask, render_template, request, jsonify, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('rf_acc_68.pkl')
scaler = joblib.load('normalizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            vals = [float(x) for x in request.form.values()]
            print("📌 Received input:", vals)

            input_data = scaler.transform([vals])
            proba = model.predict_proba(input_data)[0]  # ✅ use probabilities
            print(f"🧠 Confidence -> Healthy: {proba[0]:.2f}, Diseased: {proba[1]:.2f}")

            # ✅ Apply custom threshold
            prediction = 1 if proba[1] >= 0.75 else 0

            result_url = url_for('result', status=str(prediction), _external=True).replace("127.0.0.1", "localhost")
            return jsonify({'redirect': result_url})

        except Exception as e:
            error_url = url_for('result', status='error', msg=str(e), _external=True).replace("127.0.0.1", "localhost")
            return jsonify({'redirect': error_url})
    return render_template('index.html')

@app.route('/result')
def result():
    status = request.args.get('status')
    print("🧪 Result status =", status)

    if status == '0':
        message = "You are safe! Keep taking care of your liver health."
        advice = "Stay hydrated, eat a balanced diet, and go for routine checkups."
        risk_level = "safe"
    elif status == '1':
        message = "Liver disease detected. Please consult a doctor."
        advice = "Avoid alcohol, eat liver-friendly food, and visit your doctor regularly."
        risk_level = "risk"
    elif status == 'error':
        message = request.args.get('msg', 'An unknown error occurred.')
        advice = ""
        risk_level = "error"
    else:
        message = "Invalid status received."
        advice = ""
        risk_level = "error"

    return render_template('result.html', message=message, advice=advice, risk_level=risk_level)

if __name__ == '__main__':
    app.run(debug=True)
