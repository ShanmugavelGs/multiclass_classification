import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
import pickle
import yaml
import mlflow
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/gsshanmugavel/multiclassclassificationproject.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'gsshanmugavel'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'ddcb8cd828d5a0623d5d09fe8fd0a68608d6a2b3'

params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path):
    data = pd.read_parquet(data_path)
    x = np.array(data['Complaint'])
    y = np.array(data['Product'])
    cv = CountVectorizer()
    X = cv.fit_transform(x)

    mlflow.set_tracking_uri("https://dagshub.com/gsshanmugavel/multiclassclassificationproject.mlflow")

    # Start the MLFLOW run
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        signature = infer_signature(X_train, y_train)

        sgdmodel = SGDClassifier(loss='hinge', random_state=42)
        sgdmodel.fit(X_train, y_train)
      
        # Predict and evaluate the model
        y_pred = sgdmodel.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy score: {accuracy}")

        # Log additional metrics
        mlflow.log_metric("Accuracy", accuracy)

        # Log the Classification report
        cr = classification_report(y_test, y_pred)
        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(sgdmodel, "model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(sgdmodel, "model", signature=signature)

        # Create the dir to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path
        pickle.dump(sgdmodel, open(filename, 'wb'))

        print(f"Model saved at: {model_path}")

if __name__ == '__main__':
    train(params['data'], params['model'])