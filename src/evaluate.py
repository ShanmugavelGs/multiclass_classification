import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/gsshanmugavel/multiclassclassificationproject.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'gsshanmugavel'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'ddcb8cd828d5a0623d5d09fe8fd0a68608d6a2b3'

# Load the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
     data = pd.read_parquet(data_path)
     x = np.array(data['Complaint'])
     y = np.array(data['Product'])
     cv = CountVectorizer()
     X = cv.fit_transform(x)
     
     mlflow.set_tracking_uri("https://dagshub.com/gsshanmugavel/multiclassclassificationproject.mlflow")

     # Load the model from disk
     model = pickle.load(open(model_path, 'rb'))
     predictions = model.predict(X)
     accuracy = accuracy_score(y, predictions)
     macro_averaged_precision = metrics.precision_score(y, predictions, average = 'macro')
     micro_averaged_precision = metrics.precision_score(y, predictions, average = 'micro')
     macro_averaged_f1 = metrics.f1_score(y, predictions, average = 'macro')
     micro_averaged_f1 = metrics.f1_score(y, predictions, average = 'micro')

     # Log metrics to MLflow
     mlflow.log_metric("Accuracy", accuracy)
     mlflow.log_metric("Macro averaged Precision", macro_averaged_precision)
     mlflow.log_metric("Micro averaged Precision", micro_averaged_precision)
     mlflow.log_metric("Macro averaged f1", macro_averaged_f1)
     mlflow.log_metric("Micro averaged f1", micro_averaged_f1)

if __name__ == "__main__":
     evaluate(params["data"], params["model"])