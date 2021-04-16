from flask import Flask, request
import os
import joblib
import numpy as np
import re

app = Flask(__name__)

@app.route('/')
def helloworld():
    return 'Hello world'

@app.route('/fever', methods=['POST'])
def predict_species():
    model = joblib.load('fever.model')
    req = request.values['param']
    inputs = np.array(req.split(','), dtype=np.float64).reshape(1, -1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'ไข้หวัดไข้หวัดใหญ่'
    elif predict_target == 1:
        return 'ไข้หวัดธรรมดา'
    else:
        return 'ไข้เลือดออก'           

if __name__== '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)        