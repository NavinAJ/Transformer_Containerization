from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
import re
import transformers
from transformers import AutoTokenizer
import torch
import io
import logging

app = Flask(__name__)

def remove_pattern(input_txt,pattern):
    '''
    removes pattern from input_txt using regex
    '''
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)

    ## removes punctuations
    input_txt = re.sub(r'[^\w\s]', ' ', input_txt)

    return input_txt.strip().lower()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    logging.info("Entered Predict function")
    if request.method == 'POST':
        logging.info("POST REQUEST------------------------->")
        message = request.form['message']
        logging.info("RAW INPUT MESSAGE : ",message)
        clean_test = remove_pattern(message,"@[\w]*")
        tokenized_clean_test = clean_test.split()
        message = ' '.join(tokenized_clean_test)
        # data = [message]
        logging.info("Message passed to the model for prediuction! : ",message)
        # data = tokenizer(data,truncation=True)
        my_prediction = bert_sentiment_model(message)
        logging.info("Model Prediction :",my_prediction)
    return render_template('result.html',prediction = my_prediction[0]['label'])

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print("MODULE---------------------->",module)
        print("NAME---------------------->",name)
        if module == 'torch.storage' and name == '_load_from_bytes':
            print("Set to CPU runtime")
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else:
            print("Set to GPU runtime")
            return super().find_class(module, name)


if __name__ == '__main__':

    is_gpu_available = torch.cuda.is_available()
    if is_gpu_available:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    PATH = "model/Bert_Sentiment.pkl"

    print("Device Selected :",device)

    with open(PATH, 'rb') as f:
        buffer = io.BytesIO(f.read())
        tokenizer, bert_sentiment_model = torch.load(buffer,map_location=device)

    # torch.load(PATH, map_location=device)
	##load vectorizer and model
    # with open(PATH, 'rb') as f:
    #     tokenizer, bert_sentiment_model = CPU_Unpickler(f).load()


    print("Tokenizer : ",tokenizer)
    print("--------------------------->")
    print("BERT Model : ",bert_sentiment_model)
    
    app.run(host='0.0.0.0',port=5000,debug=True)
