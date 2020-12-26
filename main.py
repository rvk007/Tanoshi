import os

from flask import Flask, redirect, url_for, request, render_template


username = ''

app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/image_training.html', methods=["GET","POST"])
def train_image():
    if request.method == 'POST':
        username = request.form["user_name"]
        batch_size = request.form["batch_size"]
        epoch = request.form["epoch"]
        print(username,"  ", batch_size, "  ", epoch)

        username_error_message = ''
        batch_size_error_message = ''
        epoch_error_message = ''

        if (not username.isalpha()) or username == None :
            username_error_message = "Username must contain alphabets only."

        if batch_size.isnumeric():
            batch_size = int(batch_size)
            if batch_size<1 or batch_size>128 or batch_size == None:
                batch_size_error_message = 'Batch size must be between 1 and 128.'
        else:
            batch_size_error_message = 'Batch size must be a number.'

        if epoch.isnumeric():
            epoch = int(epoch)
            if epoch<1 or epoch>10 or epoch == None:
                epoch_error_message = "Number of epochs must be a number between 1 and 10."
        else:
            epoch_error_message = 'Batch size must be a number.'

        print(username_error_message,"  ", batch_size_error_message, "  ", epoch_error_message)
        return render_template("image_training.html", username_errorMessage=username_error_message, batch_size_errorMessage=batch_size_error_message, epoch_errorMessage=epoch_error_message)
    else:
        return render_template("image_training.html")


@app.route('/text_training.html')
def train_text():
    return render_template("text_training.html")


@app.route('/inference.html')
def inference():
    if request.method == 'POST':
        username = request.form["username"]
        
    popup = False
    
    if popup:
        return render_template("inference.html", popup = popup, username=username)
    else:
        return render_template("inference.html")


@app.route('/tour.html')
def tour():
    return render_template("tour.html")


@app.route('/popup.html')
def popup():
    return render_template("popup.html")


@app.route('/no_model_popup.html')
def no_model_popup():
    return render_template("no_model_popup.html")


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



