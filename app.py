import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

global model

app = Flask(__name__)


def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


def prediction(img_path):
    new_image = load_image(img_path)

    pred = model.predict(new_image)

    print(pred)

    labels = np.array(pred)
    labels[labels >= 0.6] = 1
    labels[labels < 0.6] = 0

    print(labels)
    final = np.array(labels)

    if final[0][0] == 1:
        return "Bad"
    else:
        return "Good"



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("home.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = prediction(img_path)

    return render_template("home.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    # Load the model
    model = load_model('model.h5')
    print("Model is loaded")

    # app.debug = True
    app.run(debug=True)
