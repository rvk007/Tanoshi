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


def if_training():
    config = read_from_s3()
    if config['status'] == 'active':
        return ' A model is training now. Please try again in sometime.'
    else:
        return False

def training(request, train_file, task):
    if request.method == 'POST':

        alert_message = if_training()
        if alert_message:
            print(alert_message)
            return render_template(train_file+".html", alert = alert_message)

        username = request.form["user_name"]
        model_name = request.form["modelname"]
        ratio = request.form["ratio"]
        batch_size = request.form["batch_size"]
        epoch = request.form["epoch"]
        f = request.files['dataset_file']

        if f and allowed_file(f.filename):
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

                filepath = user_name+'.txt'
                
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
                