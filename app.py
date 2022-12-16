from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
import re
import transformers
from transformers import AutoTokenizer
import torch
import io
from transformers import pipeline

app = Flask(__name__)

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    input_txt = re.sub(r'[^\w\s]', ' ', input_txt)
    return input_txt.strip().lower()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        clean_test = remove_pattern(message,"@[\w]*")
        tokenized_clean_test = clean_test.split()
        refined_message = ' '.join(tokenized_clean_test)
        my_prediction = bert_sentiment_model(refined_message)
        predicted_sentiment = MapPrediction(my_prediction[0]['label'])
        confident_score = my_prediction[0]['score']
        data = {'Sentiment':predicted_sentiment,'Score':confident_score,'Message':message}
    return render_template('result.html',prediction = data)


def MapPrediction(predicted_label):
    switcher = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "POSITIVE",
        "LABEL_2": "NEUTRAL",
        "LABEL_3": "IRRELEVANT",
    }
    return switcher[predicted_label]


# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
#         else:
#             return super().find_class(module, name)


if __name__ == '__main__':

    is_gpu_available = torch.cuda.is_available()
    if is_gpu_available:
        _device = torch.device(0)
    else:
        _device = torch.device('cpu')

    try:
        bert_sentiment_model = pipeline('text-classification',model='Bert_Sentiment_Model/',device=_device)
        print("Model loded successfully!")
    except:
        print("Unable to load the sentiment model")
    
    app.run(host='0.0.0.0',port=5000,debug=False)
