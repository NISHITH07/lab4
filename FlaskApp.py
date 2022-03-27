# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 11:45:03 2022

@author: Nishith
"""


from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


app=Flask(__name__)
pickle_in = open("classifier.pickle","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    features_value=[np.array(int_features)]
    prediction = classifier.predict(features_value)

    return render_template('index.html', prediction_text='The Species is {}'.format(prediction))
    
if __name__=='__main__':
    app.run()