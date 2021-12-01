from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html', href='static/Boxplot.jpeg')
    #return "<p>Hello, World!</p>"
    # # request_type_str = request.method
    # # if request_type_str == "GET":
    # #   return render_template('index.html', href='static/Boxplot.jpeg')
    # # else:
    # #    text = request.form['text']
    # #    return render_template('index.html', href='static/Boxplot.jpeg')


    # pkl_filename = "TrainedModel/stacking_model.pkl"
    # test_value = pd.read_csv("../../X_test.csv")[0]
    # test_input = test_value
    # with open(pkl_filename, 'rb') as file:
    #     pickle_model = pickle.load(file)
    # predict = pickle_model.predict(test_input)
    # predict_as_str = str(predict)
    # return predict_as_str


