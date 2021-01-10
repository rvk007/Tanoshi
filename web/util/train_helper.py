"""Methods used for training the dataset"""

import os
import json
import shutil
import zipfile
import pandas as pd
from random import randint
from datetime import datetime
from flask import render_template, flash

from util.s3_helper import store_to_s3, read_from_s3, put_on_s3, fetch_json, put_object, upload_file
from util.inference_helper import username_information

PREFIX = 'tanoshi'
CONFIG_PATH = 'status.json'
destination = os.path.dirname(os.path.abspath(__file__))


def if_training():
    config_data = fetch_json(CONFIG_PATH)
    config_data['status'] = 'sleeping' # to del
    if config_data['status'] == 'active':
        return ' A model is training now. Please try again in sometime.'
    else:
        return False


def training(request, train_file, task):
    alert_message = if_training()
    if alert_message:
        return render_template(f'{train_file}.html', alert=alert_message)
    else:
        username = request.form["user_name"]
        model_name = request.form["modelname"]
        ratio = request.form["ratio"]
        optimizer = request.form["optimizer"]

        try:
            is_reducelrscheduler = request.form['scheduler_toggle']
        except Exception as e:
            is_reducelrscheduler = 'off'

        patience = request.form["patience"]
        factor = request.form["factor"]
        min_lr = request.form["min_lr"]
        batch_size = request.form["batch_size"]
        learning_rate = request.form["learning_rate"]
        epoch = request.form["epoch"]
        f = request.files['dataset_file']
        original_filename = f.filename

        user_name = PREFIX + '-' + username + '-' + str(randint(0, 1000))
        # if username exists
        while username_information(user_name)[0]:
            user_name = PREFIX + '-' + username + '-' + str(randint(0, 1000))
        
        f.save('/'.join([destination, original_filename]))

        if task == 'image':
            with zipfile.ZipFile('/'.join([destination, original_filename])) as z:
                z.extractall(f'{destination}/')

            number_of_classes = 0
            f_name = original_filename.split('.')[0]
            for f in os.listdir('/'.join([destination, f_name])):
                number_of_classes+=1

            shutil.rmtree('/'.join([destination, f_name]))

            if number_of_classes==0:
                alert_message = 'The dataset entered by you is empty!'
                render_template(f'{train_file}.html', alert=alert_message)

            elif number_of_classes==1:
                alert_message = (
                    f'The dataset has only one class. Atleast two class is required to perform clasiification.')
                render_template(f'{train_file}.html', alert=alert_message)

            elif number_of_classes>10:
                alert_message = (
                    f'A maximum number of 10 classes are allowed. Your dataset has {number_of_classes} classes. Please remove some of them.')
                render_template(f'{train_file}.html', alert=alert_message)

        else:
            df = pd.DataFrame(pd.read_csv('/'.join([destination, original_filename])))
            number_of_rows = len(df)

            if number_of_rows > 10000:
                alert_message = (
                    f'The maximum number allowed rows in the .csv file is 10,000. Your dataset has {number_of_rows} rows. Please remove some of them.')
                render_template(f'{train_file}.html', alert=alert_message)

            number_of_labels = len(set(df.iloc[:, 1].values))
            if number_of_labels==1:
                alert_message = (
                    f'The dataset has only one class. Atleast two class is required to perform clasiification.')
                render_template(f'{train_file}.html', alert=alert_message)

        config_data = fetch_json(CONFIG_PATH)
        config_data['status'] = 'active'
        config_data['username'] = user_name
        put_object(CONFIG_PATH, config_data)

        if task == 'image':
            user_filename = f'{user_name}.zip'
        else:
            user_filename = f'{user_name}.csv'

        os.rename(
                '/'.join([destination, original_filename]),
                '/'.join([destination, user_filename])
        )

        if f:
            data = {
                'task': task,
                'username': user_name,
                'model': model_name,
                'ratio': int(ratio),
                'is_reducelrscheduler': is_reducelrscheduler,
                'patience': int(patience),
                'factor': float(factor),
                'min_lr': float(min_lr),
                'optimizer': optimizer,
                'batchsize': int(batch_size),
                'epochs': int(epoch),
                'learning_rate': float(learning_rate),
                'filename': user_filename,
                'training': 'started'
            }
            filepath = user_name + '.json'
            with open(filepath, 'w') as outfile:
                json.dump(data, outfile)
            upload_file(filepath, f'training/{filepath}')
            os.remove(filepath)
            
            # upload dataset
            output = upload_file(
                    source_path='/'.join([destination, user_filename]),
                    target_path=f'training/{user_filename}'
            )

            upload_message = (
                f' Username: "{user_name}". Dataset is successfully uploaded and training is in progress.'
                'It will take around 5-10 mintues to complete training.'
                'Kindly save the username for inference.'
            )
            flash(upload_message)

            # Update inference json: lambda
            config_data = fetch_json(CONFIG_PATH)
            config_data['username'] = user_name
            config_data['status'] = 'active'
            put_object(CONFIG_PATH, config_data)
            
            os.remove('/'.join([destination, user_filename]))
            return render_template(f'{train_file}.html')
        else:
            upload_message = f'  Upload Unsuccessfull. An error occured: {output[1]}. Please try again.'
            flash(upload_message)
            return render_template(f'{train_file}.html')
