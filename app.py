#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:19:43 2020

@author: proton
"""
import numpy as np
from flask import Flask , jsonify , render_template , request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl' , 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict' , methods = ['POST'])
def predict():
     int_features = [int(x) for x in request.form.values()]
     final_features = [np.array(int_features)]
     prediction = model.predict(final_features)
     
     output = round(prediction[0] , 2)
     return render_template('index.html' , predicted_text = 'The Predicted salary for the recipient is {}'.format(output))
 
    
if __name__ == '__main__':
    app.run(debug=True)