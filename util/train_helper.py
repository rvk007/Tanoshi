import os
import json
from random import randint

from flask import Flask, redirect, url_for, request, render_template 

from util.s3_helper import upload_file_to_s3, upload_localfile_to_s3, store_to_s3, read_from_s3


PREFIX = '_tanoshi'
ALLOWED_EXTENSIONS = {'zip', 'txt', 'jpeg'}

# function to check file extension
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def if_training(username):
    config = read_from_s3()
    if config['status'] == 'active':
        if config['user'] == username:
            return 'Your model is already training! Kindly wait for it to finish.'
        else:
            return 'A model is training now. Please try again in sometime.'
    else:
        return False

def training(request, username, train_file, task):
    if request.method == 'POST':

        alert_message = if_training(username)
        if alert_message:
             return render_template(train_file+".html", alert = alert_message)

        username = request.form["user_name"]
        model_name = request.form["modelname"]
        ratio = request.form["ratio"]
        batch_size = request.form["batch_size"]
        epoch = request.form["epoch"]
        f = request.files['dataset_file']

        #error messages
        error = False
        username_error_message = ''
        batch_size_error_message = ''
        epoch_error_message = ''

        if (not username.isalpha()) or username == None :
            error = True
            username_error_message = "Username must contain alphabets only."

        if batch_size.isnumeric():
            batch_size = int(batch_size)
            if batch_size<1 or batch_size>128 or batch_size == None:
                error = True
                batch_size_error_message = 'Batch size must be between 1 and 128.'
        else:
            batch_size_error_message = 'Batch size must be a number.'

        if epoch.isnumeric():
            epoch = int(epoch)
            if epoch<1 or epoch>10 or epoch == None:
                error = True
                epoch_error_message = "Number of epochs must be a number between 1 and 10."
        else:
            error =True
            epoch_error_message = 'Batch size must be a number.'

        if error:
            return render_template(train_file+".html", username_errorMessage=username_error_message, batch_size_errorMessage=batch_size_error_message, epoch_errorMessage=epoch_error_message)

        elif f and allowed_file(f.filename):
            output = upload_file_to_s3(f) 

            if output[0]:
                #dump data into a json file and push it to s3 bucket
                user_name = PREFIX + '_' + task + '_' + username + '_' + str(randint(0,1000))
                data = { 'username' : user_name,
                        'model' : model_name,
                        'ratio' : int(ratio),
                        'batchsize' : batch_size,
                        'epoch' : epoch,
                        'filename' : f.filename
                }

                filepath = os.path.join(user_name+'.txt')
                
                with open(filepath, 'w') as outfile:
                    json.dump(data, outfile)
                upload_localfile_to_s3(filepath)
                os.remove(filepath)

                popup_heading = 'Upload Successfull!'
                popup_message = 'Your dataset is successfully uploaded and training is in progress. Please wait for a while.'
                user_data = 'Your username is: '+user_name+'. Kindly save it for inferencing.'

                return render_template(train_file+".html", popup_heading=popup_heading, popup_message=popup_message, user_data=user_data)
            else:
                popup_heading = 'Upload Unsuccessfull.'
                popup_message = 'An error occured:' + output[1] +'. Please try again.'
                user_data = ''

                return render_template(train_file+".html", popup_heading=popup_heading, popup_message=popup_message, user_data=user_data)
                