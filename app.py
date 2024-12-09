# from flask import Flask, request, jsonify, render_template
# import mlflow.sklearn
# import numpy as np
# import os
#
# mlflow.set_tracking_uri(uri="http://127.0.0.1:7000")
#
# app = Flask(__name__)
#
# # Load the best model
# model = mlflow.sklearn.load_model("models:/best_model/latest")
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.is_json:
#         data = request.json['data']
#     else:
#         data = [float(request.form[f'feature_{i}']) for i in range(20)]
#
#     features = np.array(data).reshape(1, -1)
#     prediction = model.predict(features)
#     probability = model.predict_proba(features)
#
#     result = {
#         'prediction': int(prediction[0]),
#         'probability': {
#             'class_0': float(probability[0][0]),
#             'class_1': float(probability[0][1])
#         }
#     }
#
#     if request.is_json:
#         return jsonify(result)
#     else:
#         return render_template('index.html', result=result)
#
#
# if __name__ == '__main__':
#     app.run(port=5000, debug=True)




from flask import Flask, render_template, request
import mlflow
import numpy as np

app = Flask(__name__)

mlflow.set_tracking_uri(uri="http://127.0.0.1:7000")
model = mlflow.sklearn.load_model("models:/best_model/latest")
# # Load the best model
# def load_best_model():
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment_by_name("classification_experiment")
#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         order_by=["metrics.accuracy DESC"]
#     )
#     best_run = runs[0]
#     model_uri = f"runs:/{best_run.info.run_id}/model"
#     model = mlflow.sklearn.load_model(model_uri)
#     return model
#
#
# model = load_best_model()


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
    app.run(port=5000, debug=True)