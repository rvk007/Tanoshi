import os
import json 
from random import randint

from flask import Flask, redirect, url_for, request, render_template

from util.helper import upload_file_to_s3

app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))

PREFIX = '_tanoshi'
ALLOWED_EXTENSIONS = {'zip', 'txt', 'jpeg'}

# function to check file extension
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/image_training.html', methods=["GET","POST"])
def train_image():
    if request.method == 'POST':

        username = request.form["user_name"]
        model_name = request.form["modelname"]
        ratio = request.form["ratio"]
        batch_size = request.form["batch_size"]
        epoch = request.form["epoch"]
        file = request.files['dataset_file']

        print(username,"  ", model_name, "  ", ratio, " " , batch_size, "  ", epoch, " ")
        print('Filename', file)
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

        if file and allowed_file(file.filename):
            output = upload_file_to_s3(file) 
            
            # if upload success,will return file name of uploaded file
            if output:
                # write your code here 
                # to save the file name in database

                print("Success upload")

            # upload failed, redirect to upload page
            else:
                print("Unable to upload, try again")

        #dump data into a json file and push it to s3 bucket
        user_name = PREFIX + '_image_' + username + '_' + str(randint(0,1000))
        data = { 'username' : user_name,
                 'model' : model_name,
                 'ratio' : int(ratio),
                 'batchsize' : batch_size,
                 'epoch' : epoch,
                 'filename' : file
        }

        #with open(user_name + '.txt', 'w') as outfile:
        #    json.dump(data, outfile)
        # json_output = upload_file_to_s3() 
        

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



