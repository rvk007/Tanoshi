import os
import shutil
import json
from datetime import datetime

from s3 import fetch_json, put_object, delete_object, upload_file, download_file
from image_classification import train_image_classification
from sentiment_analysis import train_sentiment_analysis


STATUS_CONFIG = 'status.json'
TRAINING_CONFIG = 'training/'
INFERENCE_CONFIG = 'inference.json'
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def upload_model_data(task, username):
    files = [
        f'checkpoints/{username}_model.pt', f'{username}_accuracy_change.jpg',
    ]
    if task == 'text':
        files += [
            f'{username}_tokenizer.pkl'
        ]
    else:
        files += [
            f'{username}_correct_predictions.jpg', f'{username}_incorrect_predictions.jpg'
        ]

    for f in files:
        source = f
        if source[:10] == 'checkpoint':
            target = source[12:]
        else:
            target = f
        upload_file(
            os.path.join(DATA_PATH, source), f'inference/{target}'
        )


def get_config_data(userdata_filename):
    config_data = {}
    with open(userdata_filename, 'r') as f:
        config_data = json.load(f)

    # config_data = pickle.loads(data)
    task = config_data['task']
    username = config_data["username"]
    model_name = config_data["model"]
    ratio = config_data["ratio"]
    is_reducelrscheduler = config_data["is_reducelrscheduler"]
    patience = config_data["patience"]
    factor = config_data["factor"]
    min_lr = config_data["min_lr"]
    optimizer = config_data["optimizer"]
    batch_size = config_data["batchsize"]
    learning_rate = config_data["learning_rate"]
    epochs = config_data["epochs"]
    dataset_filename = config_data['filename']

    return (
        task, username, model_name, ratio, is_reducelrscheduler, patience, factor, min_lr, optimizer,
        batch_size, learning_rate, epochs, dataset_filename)


def main(username):
    os.makedirs(os.path.join(DATA_PATH, 'checkpoints'))

    # # Download user file
    userdata_filename = os.path.join(DATA_PATH, f'{username}.json')
    download_file(
        os.path.join(TRAINING_CONFIG, f'{username}.json'),
        userdata_filename,
    )

    (task, username, model_name, ratio, is_reducelrscheduler, patience, factor, min_lr,
        optimizer, batch_size, learning_rate, epochs, dataset_filename) = get_config_data(userdata_filename)

    # Download dataset
    download_file(
        os.path.join(TRAINING_CONFIG, dataset_filename),
        os.path.join(DATA_PATH, dataset_filename),
    )

    inference_data = {}
    if task == 'image':
        inference_data = train_image_classification(
            username, model_name, ratio, is_reducelrscheduler, patience, factor, min_lr, optimizer,
            batch_size, learning_rate, epochs, dataset_filename
        )
    elif task == 'text':
        inference_data = train_sentiment_analysis(
            username, model_name, ratio, is_reducelrscheduler, patience, factor, min_lr, optimizer,
            batch_size, learning_rate, epochs, dataset_filename
        )

    print(inference_data)
    # Upload data to S3
    upload_model_data(task, username)

    # Update inference json
    inference_config = fetch_json(INFERENCE_CONFIG)
    inference_config[username] = inference_data
    inference_config[username]['created'] = datetime.now().strftime('%d-%m-%y %H:%M')
    put_object(INFERENCE_CONFIG, inference_config)

    print(inference_config)
    # Delete data
    shutil.rmtree(DATA_PATH)

    # Delete train data from S3
    delete_object(os.path.join(TRAINING_CONFIG, dataset_filename))
    delete_object(os.path.join(TRAINING_CONFIG, f'{username}.json'))


if __name__ == '__main__':
    server_config = fetch_json(STATUS_CONFIG)
    if not server_config['dev_mode']:
        main(server_config['username'])
