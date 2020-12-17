from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/image_training.html')
def train_image():
    return render_template("image_training.html")


@app.route('/text_training.html')
def train_text():
    return render_template("text_training.html")


@app.route('/inference.html')
def inference():
    return render_template("inference.html")


@app.route('/tour.html')
def tour():
    return render_template("tour.html")

@app.route('/popup.html')
def popup():
    return render_template("popup.html")


if __name__=="__main__":
    app.run(debug=True)  #developer mode