import os
import json
from random import randint
from flask import Flask, redirect, url_for, request, render_template, flash

from util.train_helper import training, if_training, allowed_file
from util.s3_helper import upload_file_to_s3, upload_localfile_to_s3, store_to_s3, read_from_s3

username = ''
PREFIX = '_tanoshi'
app = Flask(__name__)
app.secret_key = "super secret key"
app_root = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    data = {'status' : 'sleeping',
            'user_name' : ''}
    store_to_s3(data)
    return render_template("home.html")


@app.route('/image_training.html', methods=["GET","POST"])
def train_image():
    train_file = 'image_training'
    if request.method == 'POST':
        return training(request=request, train_file=train_file, task='image')
    else:
        return render_template(train_file+".html")
                


@app.route('/text_training.html')
def train_text():
    return training(request, 'text_training', 'text')


@app.route('/inference.html')
def inference():
    if request.method == 'POST':
        username = request.form["username"]
        
    popup = True
    username = 'qwertyuiop'
    if popup:
        return render_template("inference.html", popup = popup, username=username)
    else:
        return render_template("inference.html")


@app.route('/tour.html')
def tour():
    return render_template("tour.html")


@app.route('/popup_upload.html')
def popup_upload():
    return render_template("popup_upload.html")


@app.route('/popup_training.html')
def popup_training():
    return render_template("popup_training.html")


@app.route('/popup_no_model.html')
def popup_no_model():
    return render_template("popup_no_model.html")


@app.route('/image_inference.html', methods=["GET","POST"])
def image_inference(filename=None):
    return render_template("image_inference.html")


@app.route('/text_inference.html', methods=["GET","POST"])
def text_inference():
    if request.method == 'POST':
        inputSentence = request.form["inputSentence"]
        output  = 'Positive'
        return render_template("text_inference.html", input_Sentence = inputSentence, prediction = output)
    else:
        return render_template("text_inference.html")

if __name__=="__main__":
    app.run(debug=True)  #developer mode



