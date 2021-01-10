import io
import os
import torch
import pickle
import torchtext
import spacy
from torchvision.models import resnet

from util.s3_helper import read_bucket, fetch_json, download_from_s3, download_file
from util.text_model import RNN


INFERENCE_PATH = 'inference.json'
INFERENCE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inference_cache(username, path):
    for files in os.listdir(path):
        if files == f'{username}.pt':
            return False
    return True


def username_information(username):
    config_data = fetch_json(INFERENCE_PATH)
    task = ''
    if username in config_data:
        task = config_data[username]['task_type']
        return [True, task]
    else:
        return [False, '']
  

def get_image_model(username, doesnt_exist):
    try:
        model_path = f'{INFERENCE}/static/{username}_model.pt'
        model = torch.jit.load(model_path)
        return [True, model]
    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        return [False, e]


def load_model(model_path, model_parametes):
    model = RNN(
        model_parametes['input_dim'],
        model_parametes['embedding_dim'],
        model_parametes['hidden_dim'],
        model_parametes['output_dim'],
        model_parametes['number_of_layers'],
        model_parametes['bidirectional'],
        model_parametes['dropout'],
        model_parametes['pad_index'],
        model_parametes['model_name']
    )
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def get_text_model(username, sentence, doesnt_exist):
    try:
        inference_data = fetch_json(INFERENCE_PATH)[username]
        model = load_model(f'{INFERENCE}/static/{inference_data["model_path"]}', inference_data['model_parametes'])
        tokenizer_file = open(f'{INFERENCE}/static/{inference_data["tokenizer_path"]}', 'rb')
        tokenizer = pickle.load(tokenizer_file)
        token = spacy.load('en')
        tokenized = [tok.text for tok in token.tokenizer(sentence)]
        indexed = [tokenizer[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        prediction = model(tensor, length_tensor)
        _, prediction = torch.max(prediction, 1)
        classes = inference_data['classes']
        return [True, classes[str(prediction.item())]]
    except Exception as e:
        return [False, e]


def download_inference_files(user_name):
    config_data = get_inference_data(user_name)
    task = config_data['task_type']
    files = [
        config_data['plot_path'], config_data['model_path']
    ]
    if task == 'image':
        files += [
            config_data['correct_prediction'],
            config_data['incorrect_prediction']
        ]
    else:
        files += [config_data['tokenizer_path']]
    for f in files:
        download_file(path=f'inference/{f}', target_path=f'{INFERENCE}/static/{f}')

    return config_data['classes'], config_data['accuracy']


def get_inference_data(user_name):
    config_data = fetch_json(INFERENCE_PATH)[user_name]
    return config_data


def not_exists(username):
    if os.path.exists(f'{INFERENCE_PATH}/static/{username}_model.pt'):
        return False
    return True
    