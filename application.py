import joblib
from flask import Flask, render_template, request
import numpy as np
from config.paths_config import *
from utils.helpers import Loader


app = Flask(__name__)

model = Loader.load_model(SAVED_MODEL_PATH)

FEATURES = [
        'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
        'Temp3pm', 'RainToday', 'Year', 'Month', 'Day'
    ]

LABELS = { 
        0 : 'No, it will likely not rain tomorrow.',
        1 : 'Yes, it will rain tomorrow.'
    }

@app.route("/")
def home():
    return {
        "message" : "Welcome to Australia Rain Weather Prediction App🌦️"
    }


@app.route("/predict", methods = ["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_arr = np.array(input_data).reshape(1, -1)

            pred = model.predict(input_arr)[0]
            prediction = LABELS.get(pred, 'Unknown')
    
        except Exception as e:
            print(str(e))
        
    return render_template("index.html", prediction = prediction, features = FEATURES)


if __name__ == "__main__":

    app.run(debug = True, port = 5000, host = '0.0.0.0')