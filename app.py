from flask import Flask,render_template,request
import numpy as np
# import sklearn

import joblib
# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import joblib
from tensorflow.keras.models import load_model


import os

app = Flask(__name__)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir, 'networkmodel.h5')
# pickle_file_path1 = os.path.join(my_dir, 'knncloudmodel.pkl')

# file=open(pickle_file_path ,'rb')
# model=pickle.load(file)

model=load_model(pickle_file_path)
# model=load_model('networkmodel.h5')


@app.route('/')
def index():
    tdata=np.array([[7,144,197,23.8494014,94.34814995,6.133220586,114.0512495]])
    y = model.predict(tdata)
    # print(y)
    index=np.argmax(y)
    return f"Predicted index : {index}"


# app.run()