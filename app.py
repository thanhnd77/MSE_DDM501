from flask import Flask, render_template, request
import mlflow
import numpy as np

app = Flask(__name__)

# mlflow.set_tracking_uri(uri="http://host.docker.internal:7000")
mlflow.set_tracking_uri(uri="http://127.0.0.1:7000")
model = mlflow.sklearn.load_model("models:/best_model/latest")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get features from form
        features = []
        for i in range(20):  # Assuming 20 features
            features.append(float(request.form[f'feature_{i}']))

        # Make prediction
        features = np.array(features).reshape(1, -1)

        prediction = {
            'prediction': int(model.predict(features)[0]),
            'probability': {
                'class0': float(model.predict_proba(features)[0][0]),
                'class1': float(model.predict_proba(features)[0][1])
            }
        }

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)