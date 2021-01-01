"""Methods used for training the dataset"""

import os
import pickle
from random import randint
from datetime import datetime
from flask import render_template, flash

from util.s3_helper import upload_file_to_s3, upload_localfile_to_s3, store_to_s3, read_from_s3


PREFIX = 'tanoshi'
FORMAT = '%d-%m-%Y %H:%M:'
config_filename = 'config.pkl'
ALLOWED_EXTENSIONS = {'zip', 'txt', 'jpeg', 'jpg'}


def allowed_file(filename):
    """Check file extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def if_training():
    print("33")
    config_data = read_from_s3(config_filename)
    print("11")
    config_data['status'] = 'sleeping'
    if config_data['status'] == 'active':
        return [' A model is training now. Please try again in sometime.', config_data]
    else:
        return [False, config_data]


def training(request, train_file, task):
    print("here", train_file)
    alert_message = if_training()
    print(alert_message)
    if alert_message[0]:
        print("in")
        return render_template(f'{train_file}.html', alert=alert_message[0])
    else:
        print("1 ok ")
        username = request.form["user_name"]
        model_name = request.form["modelname"]
        ratio = request.form["ratio"]
        batch_size = request.form["batch_size"]
        epoch = request.form["epoch"]
        f = request.files['dataset_file']
        # f.seek(0, os.SEEK_END)
        # file_size = f.tell()
        # print(file_size)

        user_name = PREFIX + '-' + username + '-' + str(randint(0, 1000))
        # if username exists
        f.filename = user_name + '-dataset.zip'

        if f and allowed_file(f.filename):
            print("2")
            output = upload_file_to_s3(f)
            if output[0]:
                print("3")
                # dump data into a json file and push it to s3 bucket
                data = {
                    'username': user_name,
                    'model': model_name,
                    'ratio': int(ratio),
                    'batchsize': batch_size,
                    'epochs': epoch,
                    'filename': f.filename,
                    'training': 'started'
                }

                filepath = user_name + '.txt'
                with open(filepath, 'wb') as outfile:
                    pickle.dump(data, outfile)
                upload_localfile_to_s3(filepath)
                # os.remove(filepath)
                print('4')
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
