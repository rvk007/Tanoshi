import io
import torch
import pickle
import torchtext
from torchvision.models import resnet

from util.s3_helper import read_bucket, read_from_s3, download_from_s3
from util.text_model import RNN


config_filename = 'config.pkl'


def in_bucket(filename):
    list_of_files = read_bucket()
    if filename in list_of_files:
        return True
    else:
        return False


def if_username_taken(username):
    config_data = read_from_s3(config_filename)
    if username in config_data['list_of_users']['image'].keys():
        return True
    elif username in config_data['list_of_users']['text'].keys():
        return True
    else:
        return False


def username_found(username):
    config_data = read_from_s3(config_filename)
    if username in config_data['list_of_users']['image'].keys():
        return 'image'
    elif username in config_data['list_of_users']['text'].keys():
        return 'text'
    else:
        return False


def get_image_model(username):
    try:
        # model_path = f'{username}.pt'
        model_path = 'resnet34.pt'
        download_from_s3(model_path)
        model = torch.jit.load(model_path)
        return [True, model]
    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        return [False, e]


def read_metadata(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        input_stoi = metadata['input_stoi']
        label_itos = metadata['label_itos']
    return input_stoi, label_itos


def load_model(model_path, input_stoi):
    model = RNN(
        len(set(input_stoi.values())), 100, 256, 1, 
        2, True, 0.5, input_stoi['<pad>']
    )
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def get_text_model(sentence, model_path, metadata_path):
    try:
        download_from_s3(model_path)
        download_from_s3(metadata_path)
        input_stoi, label_itos = read_metadata(metadata_path)
        model = load_model(model_path, input_stoi)
        tokenized = [tok for tok in sentence.split()]
        indexed = [input_stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor([len(indexed)])
        prediction = torch.sigmoid(model(tensor, length_tensor))

        return [True, label_itos[round(prediction.item())]]
    except Exception as e:
        return [False, e]
