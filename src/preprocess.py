import pandas as pd
import sys
import yaml
import os
import nltk
import re
from nltk.corpus import stopwords
import string

# Load the parameters from param.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["preprocess"]

# Download stopwords if not already downloaded
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

def preprocess(input_path, output_path):
    data = pd.read_parquet(input_path, columns=['Complaint', 'Product'])
    data['Complaint'] = data['Complaint'].apply(clean)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_parquet(output_path)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(
        params["input"], 
        params["output"]
        )