"""Methods used for training the dataset"""

import os
import pickle
from random import randint
from datetime import datetime
from flask import render_template, flash

from util.s3_helper import store_to_s3, read_from_s3, put_on_s3
from util.inference_helper import if_username_taken

PREFIX = 'tanoshi'
FORMAT = '%d-%m-%Y %H:%M:'
config_filename = 'config.pkl'
ALLOWED_EXTENSIONS = {'zip', 'txt', 'jpeg', 'jpg'}


def allowed_file(filename):
    """Check file extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def if_training():
    config_data = read_from_s3(config_filename)
    config_data['status'] = 'sleeping'
    if config_data['status'] == 'active':
        return [' A model is training now. Please try again in sometime.', config_data]
    else:
        return [False, config_data]


def training(request, train_file, task):
    alert_message = if_training()
    if alert_message[0]:
        return render_template(f'{train_file}.html', alert=alert_message[0])
    else:
        username = request.form["user_name"]
        model_name = request.form["modelname"]
        ratio = request.form["ratio"]
        loss_function = request.form["lossfunction"]
        optimizer = request.form["optimizer"]
        batch_size = request.form["batch_size"]
        learning_rate = request.form["learning_rate"]
        epoch = request.form["epoch"]
        f = request.files['dataset_file']

        user_name = PREFIX + '-' + username + '-' + str(randint(0, 1000))
        # if username exists
        while if_username_taken(user_name):
            user_name = PREFIX + '-' + username + '-' + str(randint(0, 1000))

        f.filename = user_name + '-dataset.zip'

        if f and allowed_file(f.filename):
            output = put_on_s3(f.filename)
            if output[0]:
                # dump data into a json file and push it to s3 bucket
                data = {
                    'task': task,
                    'username': user_name,
                    'model': model_name,
                    'ratio': int(ratio),
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'batchsize': batch_size,
                    'epochs': epoch,
                    'learning_rate': learning_rate,
                    'filename': f.filename,
                    'training': 'started'
                }

                # data = {
                #     'task': task,
                #     'username': user_name,
                #     'model': model_name,
                #     'ratio': int(ratio),
                #     'batchsize': batch_size,
                #     'epochs': epoch,
                #     'filename': f.filename,
                #     'training': 'started'
                # }

                filepath = user_name + '.txt'
                with open(filepath, 'wb') as outfile:
                    pickle.dump(data, outfile)
                put_on_s3(filepath)
                
                # will be deleted later as this has to be done by ec2

                # TESTING TEXT
                # trained_model_path = f'{user_name}.pt'
                # os.rename('resnet34.pt', trained_model_path)
                # put_on_s3(trained_model_path)
                # till here
                # os.remove(filepath)
                
                upload_message = (
                    '  Dataset is successfully uploaded and training is in progress.'
                    f' Username: " {user_name} ". Kindly save it for inferencing.'
                )
                flash(upload_message)

                # update config file
                config_data = alert_message[1]
                config_data['status'] = 'active'
                config_data['user_name'] = user_name
                config_data['list_of_users'][task][user_name] = datetime.now().strftime(FORMAT)

                store_to_s3(config_filename, config_data)

                return render_template(f'{train_file}.html')
            else:
                upload_message = f'  Upload Unsuccessfull. An error occured: {output[1]}. Please try again.'
                flash(upload_message)
                return render_template(f'{train_file}.html')
        else:
            flash('An error occured in the uploaded file. Please try again.')
            return render_template(f'{train_file}.html')
